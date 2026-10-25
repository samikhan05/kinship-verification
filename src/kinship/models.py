"""Neural models: the attention-gated tri-subject head (E2 / E2b).

Data flow for one (father, mother, child) triplet of face embeddings:

    eF, eC --> PairComparator --> h_F --+--> pairwise logit F-C   (auxiliary loss)
    eM, eC --> PairComparator --> h_M --+--> pairwise logit M-C   (auxiliary loss)
                 (shared weights)       |
                                        v
              attention gate:  alpha = softmax(W [h_F ; h_M])  ->  (alpha_F, alpha_M)
              fusion:          z = alpha_F * h_F + alpha_M * h_M
              add:             cos((eF + eM) / 2, eC)          "mixed parents" similarity
                               cos(eF, eC), cos(eM, eC)        skip connections (see below)
                                        v
              classifier MLP  -->  triplet logit  -->  P(child of this couple)

* The comparator is SHARED ("Siamese"): father-child and mother-child are judged
  by the same network, so both parents are compared in the same way.
* alpha says how much the final decision relies on each parent. That gives the
  interpretable "father 62% / mother 38%" output.
* The auxiliary pairwise losses force h_F and h_M to carry real parent-child
  evidence on their own, which makes the attention weights meaningful.

Changes after the first Kaggle run (2026-10-25). Without them the head overfit:
best validation AUC at epoch 1, then worse while training loss kept falling.
  * A shared linear projection (512 -> proj_dim) before the comparison features
    cuts the first layer from ~400k to ~65k weights.
  * The raw cosine similarities go straight to the classifier ("skip
    connections"): the model starts from what the strong E0 baseline knows and
    only has to learn what to add on top.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def pair_features(p: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """[|p - c|, p * c, (p - c)^2, cos(p, c)] - standard symmetric comparison features.

    |p - c| and (p - c)^2 measure per-dimension differences, p * c measures
    per-dimension agreement, and cos(p, c) is the overall similarity."""
    diff = p - c
    cos = F.cosine_similarity(p, c, dim=-1).unsqueeze(-1)
    return torch.cat([diff.abs(), p * c, diff * diff, cos], dim=-1)


class PairComparator(nn.Module):
    """Shared parent-child comparator: embeddings of a parent and a child -> h (vector)."""

    def __init__(self, emb_dim: int = 512, proj_dim: int = 128, hidden: int = 128,
                 out_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(emb_dim, proj_dim)  # shared: parent and child go through the same layer
        self.net = nn.Sequential(
            nn.Linear(3 * proj_dim + 1, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.ReLU())
        self.pair_logit = nn.Linear(out_dim + 1, 1)   # "is this parent related to this child?"

    def forward(self, p: torch.Tensor, c: torch.Tensor):
        """Returns (h, pairwise logit, raw cosine similarity)."""
        cos_raw = F.cosine_similarity(p, c, dim=-1).unsqueeze(-1)
        pp, cc = F.normalize(self.proj(p), dim=-1), F.normalize(self.proj(c), dim=-1)
        h = self.net(pair_features(pp, cc))
        return h, self.pair_logit(torch.cat([h, cos_raw], dim=-1)).squeeze(-1), cos_raw


class TriSubjectHead(nn.Module):
    """Attention-gated tri-subject kinship head (main model, E2)."""

    def __init__(self, emb_dim: int = 512, proj_dim: int = 128, hidden: int = 128,
                 pair_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.comparator = PairComparator(emb_dim, proj_dim, hidden, pair_dim, dropout)
        self.gate = nn.Linear(2 * pair_dim, 2)    # sees both parents, outputs one score per parent
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim + 3, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, eF: torch.Tensor, eM: torch.Tensor, eC: torch.Tensor) -> dict[str, torch.Tensor]:
        h_f, logit_fc, cos_fc = self.comparator(eF, eC)
        h_m, logit_mc, cos_mc = self.comparator(eM, eC)
        alpha = torch.softmax(self.gate(torch.cat([h_f, h_m], dim=-1)), dim=-1)   # (B, 2)
        z = alpha[:, :1] * h_f + alpha[:, 1:] * h_m
        mixed = F.cosine_similarity((eF + eM) / 2, eC, dim=-1).unsqueeze(-1)
        logit = self.classifier(torch.cat([z, mixed, cos_fc, cos_mc], dim=-1)).squeeze(-1)
        return {"logit": logit, "logit_fc": logit_fc, "logit_mc": logit_mc,
                "alpha_f": alpha[:, 0], "alpha_m": alpha[:, 1]}


def tri_subject_loss(out: dict[str, torch.Tensor], y: torch.Tensor, y_fc: torch.Tensor,
                     y_mc: torch.Tensor, aux_weight: float = 0.3) -> torch.Tensor:
    """BCE on the triplet label + aux_weight * (BCE on F-C + BCE on M-C pair labels).

    Pair labels differ from the triplet label only for hard negatives: in
    (F', M, C) the mother-child pair is still a true pair (y_mc = 1)."""
    bce = F.binary_cross_entropy_with_logits
    return bce(out["logit"], y) + aux_weight * (bce(out["logit_fc"], y_fc) + bce(out["logit_mc"], y_mc))


# --------------------------------------------------------------------------- E3: fine-tuning
def facenet_backbone(pretrained: str | None = "vggface2", unfreeze=("block8", "last_linear", "last_bn")):
    """InceptionResnetV1 (facenet_pytorch) with only the listed top-level blocks trainable.

    Why only the last block(s): with ~1,400 training families, updating all ~23M
    weights would quickly overfit and could destroy the general face features
    learned from VGGFace2. The early layers (edges, textures, face parts) stay as
    they are; only the final, most task-specific layers adapt to kinship.
    """
    from facenet_pytorch import InceptionResnetV1

    net = InceptionResnetV1(pretrained=pretrained)
    for name, p in net.named_parameters():
        p.requires_grad = any(name == u or name.startswith(u + ".") for u in unfreeze)
    return net


class FineTuneModel(nn.Module):
    """E3: FaceNet backbone (last blocks unfrozen) + the E2 tri-subject head, trained end to end."""

    def __init__(self, backbone: nn.Module, head: TriSubjectHead, unfreeze=("block8", "last_linear", "last_bn")):
        super().__init__()
        self.backbone, self.head, self.unfreeze = backbone, head, list(unfreeze)

    def train(self, mode: bool = True) -> "FineTuneModel":
        super().train(mode)
        if mode:
            # Frozen layers stay in eval mode: their BatchNorm statistics must not
            # drift with our small batches, and the backbone's dropout stays off.
            for name, module in self.backbone.named_children():
                if name not in self.unfreeze:
                    module.eval()
        return self

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Face crops (normalised, N x 3 x 160 x 160) -> L2-normalised 512-d embeddings."""
        return F.normalize(self.backbone(x).float(), dim=-1)

    def forward(self, xF: torch.Tensor, xM: torch.Tensor, xC: torch.Tensor) -> dict[str, torch.Tensor]:
        n = xF.shape[0]
        e = self.embed(torch.cat([xF, xM, xC]))       # one backbone pass for all three people
        return self.head(e[:n], e[n:2 * n], e[2 * n:])
