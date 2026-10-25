"""Training loops: the tri-subject head on cached embeddings (E2 / E2b).
Backbone fine-tuning (E3) is added in M4.

Each epoch is built fresh:
  * every kin triplet of the training split, plus ONE newly drawn negative for it:
      main negative  (F, M, C')  C' = a child of another family, same gender as C
      hard negative  (F', M, C) or (F, M', C), with probability `hard_negative_fraction`
  * every person is represented by the mean of a RANDOM SUBSET of their images
    (sometimes one image, sometimes all), so the model sees new inputs every
    epoch and gets used to both single images and templates;
  * the child's images always come from photos that neither parent's chosen
    images come from, so the model cannot learn the same-photo shortcut.
Early stopping on validation ROC-AUC (headline protocol: templates, different photos).
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .align import l2_normalise
from .embed import EmbeddingStore
from .evaluate import choose_threshold, keep_complete_pairs, triplet_embeddings
from .models import TriSubjectHead, tri_subject_loss


class EpochSampler:
    """Draws a fresh training epoch of (eF, eM, eC, y, y_fc, y_mc) arrays."""

    def __init__(self, store: EmbeddingStore, positives: pd.DataFrame, persons: pd.DataFrame,
                 hard_fraction: float = 0.0):
        self.store, self.hard_fraction = store, hard_fraction
        self.pos = positives[["F", "M", "C"]].itertuples(index=False, name=None)
        self.pos = [t for t in self.pos if all(p in store for p in t)]
        info = persons.set_index("pid")
        self.fid = info["fid"].to_dict()
        self.gender = info["gender"].to_dict()

        def pool(pids):
            """{gender: array of pids} for candidate replacements."""
            pids = sorted({p for p in pids if p in store and isinstance(self.gender.get(p), str)})
            return {g: np.array([p for p in pids if self.gender[p] == g], dtype=object) for g in ("m", "f")}

        self.children = pool(t[2] for t in self.pos)              # stand-ins for C
        self.adults = pool(x for t in self.pos for x in t[:2])    # stand-ins for F / M
        self._rows: dict[tuple, tuple | None] = {}

    def _stranger(self, pool: dict, like: str, avoid_fids: set, rng) -> str | None:
        cands = pool.get(self.gender.get(like), np.array([], dtype=object))
        for _ in range(20):                                       # rejection sampling
            if len(cands) == 0:
                return None
            p = cands[rng.integers(len(cands))]
            if self.fid[p] not in avoid_fids:
                return p
        return None

    def _subset_mean(self, rows: np.ndarray, rng) -> np.ndarray:
        k = int(rng.integers(1, len(rows) + 1))
        return l2_normalise(self.store.emb[rng.choice(rows, size=k, replace=False)].mean(axis=0))

    def items(self, rng: np.random.Generator) -> list[tuple]:
        """One epoch of training items: (rows_F, rows_M, rows_C, y, y_FC, y_MC).

        rows_* are the image rows each person may use (child photos never shared
        with the parents' chosen photos); y is the triplet label, y_FC / y_MC the
        pair labels used by the auxiliary losses."""
        drawn = []
        for f, m, c in self.pos:
            drawn.append((f, m, c, 1.0, 1.0, 1.0))
            fams = {self.fid[f], self.fid[m], self.fid[c]}
            if rng.random() < self.hard_fraction:                 # hard: swap one parent
                if rng.random() < 0.5:
                    f2 = self._stranger(self.adults, f, fams, rng)
                    if f2 is not None:
                        drawn.append((f2, m, c, 0.0, 0.0, 1.0))
                else:
                    m2 = self._stranger(self.adults, m, fams, rng)
                    if m2 is not None:
                        drawn.append((f, m2, c, 0.0, 1.0, 0.0))
            else:                                                 # main: swap the child
                c2 = self._stranger(self.children, c, fams, rng)
                if c2 is not None:
                    drawn.append((f, m, c2, 0.0, 0.0, 0.0))
        out = []
        for f, m, c, y, yf, ym in drawn:
            key = (f, m, c)
            if key not in self._rows:
                self._rows[key] = self.store.triplet_rows(f, m, c)
            if self._rows[key] is not None:
                out.append((*self._rows[key], y, yf, ym))
        return out

    def epoch(self, rng: np.random.Generator):
        """Embedding arrays for one epoch: each person = mean of a random subset of
        their allowed images (E2 / E2b)."""
        eF, eM, eC, ys = [], [], [], []
        for rf, rm, rc, y, yf, ym in self.items(rng):
            eF.append(self._subset_mean(rf, rng))
            eM.append(self._subset_mean(rm, rng))
            eC.append(self._subset_mean(rc, rng))
            ys.append((y, yf, ym))
        ys = np.array(ys, np.float32)
        return np.stack(eF), np.stack(eM), np.stack(eC), ys[:, 0], ys[:, 1], ys[:, 2]


class HeadScorer:
    """Wraps a trained TriSubjectHead so evaluate_triplet_scorer can use it."""

    def __init__(self, model: TriSubjectHead, device: str = "cpu"):
        self.model, self.device = model.eval(), device

    @torch.no_grad()
    def score_details(self, eF, eM, eC):
        outs = []
        for i in range(0, len(eF), 1024):
            t = [torch.as_tensor(x[i:i + 1024], dtype=torch.float32, device=self.device) for x in (eF, eM, eC)]
            o = self.model(*t)
            outs.append({k: v.float().cpu().numpy() for k, v in o.items()})
        cat = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
        sig = lambda x: 1 / (1 + np.exp(-x))
        return sig(cat["logit"]), {"alpha_F": cat["alpha_f"], "alpha_M": cat["alpha_m"],
                                   "p_FC": sig(cat["logit_fc"]), "p_MC": sig(cat["logit_mc"])}

    def score(self, eF, eM, eC):
        return self.score_details(eF, eM, eC)[0]


def _val_arrays(store: EmbeddingStore, df: pd.DataFrame):
    """Fixed validation inputs: templates from different photos (headline protocol)."""
    *emb, ok = triplet_embeddings(store, df, "template", cross_photo=True)
    keep = keep_complete_pairs(df, ok)
    return [e[keep] for e in emb], df["label"].to_numpy()[keep]


def pretrain_comparator(model: TriSubjectHead, store: EmbeddingStore, pairs_train: pd.DataFrame,
                        pairs_val: pd.DataFrame | None, hp: dict, rng: np.random.Generator,
                        device: str, log) -> list[dict]:
    """Stage 1 of E2-pairs: train only the shared parent-child comparator on the
    pairwise data (all related pairs + parent-child pairs vs strangers).

    The pairwise training split has ~8x more examples and many more people than
    the 1,434 kin triplets, so the comparator learns "kin vs not kin" from far more
    families before the triplet head is trained on top of it. Images of the two
    people always come from different photos.
    """
    pairs = []
    for a, b, y in zip(pairs_train["p1"], pairs_train["p2"], pairs_train["label"]):
        if a in store and b in store:
            rows = store.disjoint_rows(a, b)
            if rows is not None:
                pairs.append((rows[0], rows[1], float(y)))
    val = None
    if pairs_val is not None:
        ea, eb, ok = store.pair(pairs_val["p1"], pairs_val["p2"], "template", cross_photo=True)
        val = (ea[ok], eb[ok], pairs_val["label"].to_numpy()[ok])
    comp = model.comparator
    opt = torch.optim.AdamW(comp.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    bs, history = int(hp["batch_size"]), []
    for epoch in range(1, int(hp["pretrain_pair_epochs"]) + 1):
        comp.train()
        perm = rng.permutation(len(pairs))
        total = 0.0
        for i in range(0, len(perm), bs):
            batch = [pairs[j] for j in perm[i:i + bs]]
            subset = lambda rows: l2_normalise(store.emb[rng.choice(rows, size=int(rng.integers(1, len(rows) + 1)),
                                                                    replace=False)].mean(axis=0))
            a = torch.as_tensor(np.stack([subset(r[0]) for r in batch]), device=device)
            b = torch.as_tensor(np.stack([subset(r[1]) for r in batch]), device=device)
            y = torch.as_tensor([r[2] for r in batch], dtype=torch.float32, device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(comp(a, b)[1], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        row = {"pretrain_epoch": epoch, "loss": total / max(len(perm), 1)}
        if val is not None:
            comp.eval()
            with torch.no_grad():
                logits = comp(torch.as_tensor(val[0], device=device), torch.as_tensor(val[1], device=device))[1]
            row["val_pair_auc"] = roc_auc_score(val[2], logits.float().cpu().numpy())
        history.append(row)
        log.info(f"pair pretraining epoch {epoch}: loss {row['loss']:.4f}  val pair AUC {row.get('val_pair_auc', float('nan')):.4f}")
    return history


def train_head(store: EmbeddingStore, train_df: pd.DataFrame, val_df: pd.DataFrame,
               persons: pd.DataFrame, hp: dict, seed: int, log, log_csv: str | Path,
               ckpt_path: str | Path, val_hard_df: pd.DataFrame | None = None,
               device: str | None = None, pairs_train: pd.DataFrame | None = None,
               pairs_val: pd.DataFrame | None = None) -> tuple[TriSubjectHead, dict]:
    """Train E2 with AdamW + cosine LR, early stopping on validation ROC-AUC.
    If hp["pretrain_pair_epochs"] > 0 and pairs_train is given, the comparator is
    first pretrained on pairwise data (`pretrain_comparator`).
    Saves the best checkpoint and a per-epoch CSV log. Returns (best model, summary)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sampler = EpochSampler(store, train_df[train_df["label"] == 1], persons, float(hp["hard_negative_fraction"]))
    (vF, vM, vC), v_y = _val_arrays(store, val_df)
    hard = _val_arrays(store, val_hard_df) if val_hard_df is not None else None

    model_args = {"emb_dim": int(store.emb.shape[1]), "proj_dim": int(hp["proj_dim"]),
                  "hidden": int(hp["hidden"]), "pair_dim": int(hp["pair_dim"]),
                  "dropout": float(hp["dropout"])}
    model = TriSubjectHead(**model_args).to(device)
    pretrain_history = []
    if int(hp.get("pretrain_pair_epochs", 0)) > 0 and pairs_train is not None:
        pretrain_history = pretrain_comparator(model, store, pairs_train, pairs_val, hp, rng, device, log)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    epochs, bs = int(hp["epochs"]), int(hp["batch_size"])
    steps = epochs * math.ceil(2 * len(sampler.pos) / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    scorer = HeadScorer(model, device)

    best = {"val_auc": -1.0, "epoch": 0}
    wait, history = 0, []
    Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        eF, eM, eC, y, yf, ym = sampler.epoch(rng)
        perm = rng.permutation(len(y))
        total = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            t = [torch.as_tensor(a[idx], device=device) for a in (eF, eM, eC, y, yf, ym)]
            loss = tri_subject_loss(model(t[0], t[1], t[2]), t[3], t[4], t[5], float(hp["aux_weight"]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)

        model.eval()
        v_s = scorer.score(vF, vM, vC)
        row = {"epoch": epoch, "train_loss": total / len(perm), "train_items": len(perm),
               "val_auc": roc_auc_score(v_y, v_s),
               "val_acc": float(((v_s >= choose_threshold(v_y, v_s)) == v_y).mean()),
               "val_hard_auc": roc_auc_score(hard[1], scorer.score(*hard[0])) if hard else float("nan"),
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        improved = row["val_auc"] > best["val_auc"] + 1e-4
        if improved:
            best = {"val_auc": row["val_auc"], "epoch": epoch}
            torch.save({"state_dict": model.state_dict(), "model_args": model_args, "epoch": epoch,
                        "val_auc": row["val_auc"], "hyperparams": hp}, ckpt_path)
            wait = 0
        else:
            wait += 1
        log.info(f"epoch {epoch:2d}  loss {row['train_loss']:.4f}  val AUC {row['val_auc']:.4f}  "
                 f"val acc {row['val_acc']:.3f}  val hard AUC {row['val_hard_auc']:.4f}"
                 + ("  *best*" if improved else ""))
        if wait >= int(hp["patience"]):
            log.info(f"Early stopping: no val AUC improvement for {wait} epochs")
            break

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    model = load_head(ckpt_path, device)
    return model, {"best_epoch": best["epoch"], "best_val_auc": best["val_auc"],
                   "epochs_run": len(history), "model_args": model_args, "hyperparams": hp,
                   "pair_pretraining": pretrain_history}


def load_head(path: str | Path, device: str = "cpu") -> TriSubjectHead:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TriSubjectHead(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


# --------------------------------------------------------------------------- analysis
def alpha_analysis(store: EmbeddingStore, test: pd.DataFrame, scorer: HeadScorer) -> dict:
    """How the attention weights behave on KIN test triplets (headline protocol).

    If alpha_F really means "the child resembles the father more", it should rise
    with cos(F,C) - cos(M,C). Reported: Spearman correlation, and how often alpha
    points to the more similar parent. Decides whether the demo may show alpha as
    "resembles father X% / mother Y%".
    """
    from scipy.stats import spearmanr

    from .evaluate import rowwise_cos

    pos = test[test["label"] == 1].reset_index(drop=True)
    *emb, ok = triplet_embeddings(store, pos, "template", cross_photo=True)
    eF, eM, eC = (e[ok] for e in emb)
    pos = pos[ok].reset_index(drop=True)
    _, d = scorer.score_details(eF, eM, eC)
    cos_gap = rowwise_cos(eF, eC) - rowwise_cos(eM, eC)          # > 0: child closer to father
    rho, pval = spearmanr(d["alpha_F"], cos_gap)
    out = {"n_kin_triplets": int(len(pos)),
           "spearman_alphaF_vs_cos_gap": float(rho), "spearman_p_value": float(pval),
           "share_where_alpha_picks_more_similar_parent": float(np.mean((d["alpha_F"] > 0.5) == (cos_gap > 0))),
           "mean_alpha_F": float(d["alpha_F"].mean()),
           "share_alpha_F_above_half": float((d["alpha_F"] > 0.5).mean())}
    for kind in ("FMS", "FMD"):
        m = (pos["kind"] == kind).to_numpy()
        if m.any():
            out[f"mean_alpha_F_{kind}"] = float(d["alpha_F"][m].mean())
    return out


# --------------------------------------------------------------------------- E3: fine-tuning
def augment(x: torch.Tensor) -> torch.Tensor:
    """Light augmentation of face crops (N x 3 x H x W, pixel values 0-255):
    random horizontal flip and +-10% brightness / contrast. No rotations or crops,
    because the face must stay where the detector put it."""
    n = x.shape[0]
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    contrast = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2
    brightness = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return (((x - mean) * contrast + mean) * brightness).clamp(0, 255)


def _to_input(crops_uint8: np.ndarray, device: str, aug: bool = False) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(crops_uint8)).to(device).permute(0, 3, 1, 2).float()
    if aug:
        x = augment(x)
    return (x - 127.5) / 128.0                       # facenet_pytorch's input normalisation


@torch.no_grad()
def embed_rows(model, crops: np.ndarray, rows: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    """Embeddings of crops[rows] (in the given order) with the (fine-tuned) backbone."""
    model.eval()
    out = []
    for i in range(0, len(rows), batch):
        x = _to_input(crops[rows[i:i + batch]], device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            out.append(model.embed(x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def finetune_e3(store: EmbeddingStore, index: pd.DataFrame, crops: np.ndarray, head: TriSubjectHead,
                train_df: pd.DataFrame, val_df: pd.DataFrame, persons: pd.DataFrame, ft: dict, seed: int,
                log, log_csv: str | Path, ckpt_path: str | Path, val_hard_df: pd.DataFrame | None = None,
                pretrained: str | None = "vggface2", device: str | None = None):
    """E3: fine-tune FaceNet's last block(s) together with the tri-subject head.

    * Start: backbone = FaceNet (VGGFace2), head = the E2b head trained on frozen
      FaceNet embeddings, so E3 vs E2b isolates the effect of fine-tuning.
    * Two learning rates (AdamW + cosine schedule): small for the pretrained
      backbone (~1e-5, so it changes gently), larger for the head (~1e-3).
    * Each step: a batch of triplets, one random allowed image per person (same
      sampling and photo rules as E2), light augmentation, mixed precision (fp16)
      on GPU, optional gradient accumulation.
    * After each epoch: embed all validation images, score the validation triplets
      with templates from different photos, early stopping on validation ROC-AUC.
    Returns (best model, summary).
    """
    from .models import FineTuneModel, facenet_backbone

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(ft.get("amp", True)) and device == "cuda"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    unfreeze = list(ft["unfreeze"])
    model = FineTuneModel(facenet_backbone(pretrained, unfreeze), head, unfreeze).to(device)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": float(ft["lr_backbone"])},
                             {"params": model.head.parameters(), "lr": float(ft["lr_head"])}],
                            weight_decay=float(ft["weight_decay"]))
    sampler = EpochSampler(store, train_df[train_df["label"] == 1], persons, float(ft["hard_negative_fraction"]))
    bs, accum, epochs = int(ft["batch_triplets"]), int(ft["accum_steps"]), int(ft["epochs"])
    steps = epochs * math.ceil(2 * len(sampler.pos) / (bs * accum))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Validation: all images of the people in the validation tables.
    val_people = set(val_df[["F", "M", "C"]].to_numpy().ravel())
    if val_hard_df is not None:
        val_people |= set(val_hard_df[["F", "M", "C"]].to_numpy().ravel())
    val_rows = np.flatnonzero(index["pid"].isin(val_people).to_numpy())

    def validate():
        emb = np.zeros((len(index), store.emb.shape[1]), np.float32)
        emb[val_rows] = embed_rows(model, crops, val_rows, device)
        vstore = EmbeddingStore(emb, index)
        scorer = HeadScorer(model.head, device)
        (vF, vM, vC), vy = _val_arrays(vstore, val_df)
        vs = scorer.score(vF, vM, vC)
        res = {"val_auc": roc_auc_score(vy, vs), "val_acc": float(((vs >= choose_threshold(vy, vs)) == vy).mean())}
        if val_hard_df is not None:
            (hF, hM, hC), hy = _val_arrays(vstore, val_hard_df)
            res["val_hard_auc"] = roc_auc_score(hy, scorer.score(hF, hM, hC))
        return res

    start = validate()
    log.info(f"before fine-tuning: val AUC {start['val_auc']:.4f}  val acc {start['val_acc']:.3f}")
    best = {"val_auc": start["val_auc"], "epoch": 0}
    torch.save({"state_dict": model.state_dict(), "head_args": _head_args(head), "unfreeze": unfreeze,
                "epoch": 0, "val_auc": start["val_auc"], "hyperparams": ft}, ckpt_path)
    history, wait = [{"epoch": 0, **start}], 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        items = sampler.items(rng)
        perm = rng.permutation(len(items))
        total, n_done = 0.0, 0
        opt.zero_grad()
        for step, i in enumerate(range(0, len(perm), bs)):
            batch = [items[j] for j in perm[i:i + bs]]
            rows = np.array([[rng.choice(b[k]) for b in batch] for k in range(3)])     # (3, B) image rows
            x = _to_input(crops[rows.ravel()], device, aug=True)
            y, yf, ym = (torch.tensor([b[k] for b in batch], dtype=torch.float32, device=device) for k in (3, 4, 5))
            n = len(batch)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(x[:n], x[n:2 * n], x[2 * n:])
            out = {k: v.float() for k, v in out.items()}
            loss = tri_subject_loss(out, y, yf, ym, float(ft["aux_weight"])) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0 or i + bs >= len(perm):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            total += loss.item() * accum * n
            n_done += n
        row = {"epoch": epoch, "train_loss": total / max(n_done, 1), **validate(),
               "lr_backbone": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        improved = row["val_auc"] > best["val_auc"] + 1e-4
        if improved:
            best = {"val_auc": row["val_auc"], "epoch": epoch}
            torch.save({"state_dict": model.state_dict(), "head_args": _head_args(head), "unfreeze": unfreeze,
                        "epoch": epoch, "val_auc": row["val_auc"], "hyperparams": ft}, ckpt_path)
            wait = 0
        else:
            wait += 1
        log.info(f"epoch {epoch:2d}  loss {row['train_loss']:.4f}  val AUC {row['val_auc']:.4f}  "
                 f"val acc {row['val_acc']:.3f}  val hard AUC {row.get('val_hard_auc', float('nan')):.4f}  "
                 f"({row['seconds']:.0f}s)" + ("  *best*" if improved else ""))
        if wait >= int(ft["patience"]):
            log.info(f"Early stopping: no val AUC improvement for {wait} epochs")
            break

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in history for k in r)))
        w.writeheader()
        w.writerows(history)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {"best_epoch": best["epoch"], "best_val_auc": best["val_auc"],
                   "val_auc_before_finetuning": start["val_auc"], "epochs_run": len(history) - 1,
                   "trainable_backbone_params": int(sum(p.numel() for p in backbone_params)),
                   "hyperparams": ft}


def _head_args(head: TriSubjectHead) -> dict:
    c = head.comparator
    return {"emb_dim": c.proj.in_features, "proj_dim": c.proj.out_features, "hidden": c.net[0].out_features,
            "pair_dim": c.net[3].out_features, "dropout": c.net[2].p}
