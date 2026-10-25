"""Model forward shapes, the loss, and a short training run on synthetic data."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kinship.models import TriSubjectHead, pair_features, tri_subject_loss  # noqa: E402


def _batch(n=8, d=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1) for _ in range(3)]


def test_pair_features_shape_and_symmetry():
    p, c, _ = _batch()
    f = pair_features(p, c)
    assert f.shape == (8, 3 * 512 + 1)
    assert torch.allclose(f, pair_features(c, p))            # comparing p to c == comparing c to p


def test_head_forward_shapes_and_attention():
    model = TriSubjectHead().eval()
    out = model(*_batch())
    for key in ("logit", "logit_fc", "logit_mc", "alpha_f", "alpha_m"):
        assert out[key].shape == (8,)
    assert torch.allclose(out["alpha_f"] + out["alpha_m"], torch.ones(8))
    assert ((out["alpha_f"] >= 0) & (out["alpha_f"] <= 1)).all()


def test_loss_backpropagates_to_every_part():
    model = TriSubjectHead()
    y = torch.tensor([1.0, 0.0] * 4)
    loss = tri_subject_loss(model(*_batch()), y, y, torch.ones(8))
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss)
    for name in ("comparator.net.0.weight", "comparator.pair_logit.weight", "gate.weight", "classifier.0.weight"):
        assert dict(model.named_parameters())[name].grad.abs().sum() > 0, name


def test_training_learns_planted_signal(kaggle_synth, tmp_path):
    """3 quick epochs on SmokeTestAligner embeddings of the synthetic data."""
    from kinship.align import SmokeTestAligner
    from kinship.embed import EmbeddingStore, extract_embeddings
    from kinship.train import HeadScorer, load_head, train_head
    from kinship.triplets import build_all_splits
    from kinship.utils import get_logger

    ds, cfg = kaggle_synth["ds"], kaggle_synth["cfg"]
    emb, index = extract_embeddings(ds, SmokeTestAligner(cfg), tmp_path / "emb", chunk_size=1000)
    store = EmbeddingStore(emb, index)
    out = build_all_splits(ds, cfg)
    persons = pd.DataFrame([{"pid": p.pid, "fid": p.fid, "gender": p.gender} for p in ds.persons.values()])
    hp = {**cfg["train"]["head"], "epochs": 3, "hard_negative_fraction": 0.5}
    model, summary = train_head(store, out["triplets"]["train"], out["triplets"]["val"], persons, hp, 0,
                                get_logger("test"), tmp_path / "log.csv", tmp_path / "head.pt",
                                out["hard"]["val"], device="cpu")
    assert summary["epochs_run"] == 3 and (tmp_path / "log.csv").exists()
    assert summary["best_val_auc"] > 0.6                       # well above chance (0.5)
    reloaded = load_head(tmp_path / "head.pt")
    e = [np.asarray(x) for x in _batch(4)]
    s, details = HeadScorer(reloaded).score_details(*e)
    assert s.shape == (4,) and set(details) == {"alpha_F", "alpha_M", "p_FC", "p_MC"}
    assert np.allclose(details["alpha_F"] + details["alpha_M"], 1, atol=1e-5)
