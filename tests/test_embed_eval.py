"""Embedding caching / resuming, metrics, and the E0 / E1 evaluation on synthetic data.

Uses SmokeTestAligner (no InsightFace needed). Synthetic children inherit a mix
of their parents' colours, so a correct pipeline must score kin above non-kin.
"""
import numpy as np
import pandas as pd
import pytest

from kinship.align import SmokeTestAligner
from kinship.embed import EmbeddingStore, extract_embeddings, image_list, load_embeddings
from kinship.evaluate import (ClassicalScorer, CosineScorer, choose_threshold, compute_metrics,
                              evaluate_pairwise_cosine, evaluate_triplet_scorer,
                              handcrafted_features, tar_at_far, triplet_embeddings)
from kinship.triplets import build_all_splits


class CountingAligner(SmokeTestAligner):
    """Counts calls, so we can check that cached chunks are not recomputed."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls = 0

    def process(self, img):
        self.calls += 1
        return super().process(img)


# ---------------------------------------------------------------------- metrics
def test_threshold_and_metrics():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = choose_threshold(y, s)
    assert 0.3 < t < 0.7
    m = compute_metrics(y, s, t)
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0 and m["f1"] == 1.0
    assert tar_at_far(y, s, 0.1) == 1.0
    # Inverted scores: AUC 0; the best threshold then predicts everything as one class.
    assert compute_metrics(y, -s, choose_threshold(y, -s))["roc_auc"] == 0.0
    # Ties: the threshold may not split equal scores.
    y2, s2 = np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])
    assert compute_metrics(y2, s2, choose_threshold(y2, s2))["accuracy"] == 0.5


# ---------------------------------------------------------------------- extraction
@pytest.fixture(scope="module")
def extracted(kaggle_synth, tmp_path_factory):
    out = tmp_path_factory.mktemp("emb")
    aligner = CountingAligner(kaggle_synth["cfg"])
    emb, index = extract_embeddings(kaggle_synth["ds"], aligner, out, chunk_size=100)
    return {"out": out, "emb": emb, "index": index, "calls": aligner.calls}


def test_extraction_shapes(kaggle_synth, extracted):
    n = len(image_list(kaggle_synth["ds"]))
    assert extracted["emb"].shape == (n, 512)
    assert extracted["calls"] == n
    assert np.allclose(np.linalg.norm(extracted["emb"], axis=1), 1, atol=1e-5)
    emb, index = load_embeddings(extracted["out"])
    assert len(index) == n and index["image"].tolist() == [i for i, _ in image_list(kaggle_synth["ds"])]


def test_extraction_resumes(kaggle_synth, extracted):
    out = extracted["out"]
    (out / "chunk_00001.npy").unlink()                 # simulate a crash in chunk 1
    aligner = CountingAligner(kaggle_synth["cfg"])
    emb, _ = extract_embeddings(kaggle_synth["ds"], aligner, out, chunk_size=100)
    assert aligner.calls == 100                        # only the missing chunk was redone
    assert np.allclose(emb, extracted["emb"])


# ---------------------------------------------------------------------- experiments
@pytest.fixture(scope="module")
def eval_setup(kaggle_synth, extracted):
    out = build_all_splits(kaggle_synth["ds"], kaggle_synth["cfg"])
    store = EmbeddingStore(extracted["emb"], extracted["index"])
    sets = {"val": out["triplets"]["val"], "test": out["triplets"]["test"], "test_hard": out["hard"]["test"]}
    return out, store, sets


def test_e0_finds_planted_signal(eval_setup, tmp_path):
    out, store, sets = eval_setup
    res = evaluate_triplet_scorer("E0-mean", "test", CosineScorer("mean"), store, sets, tmp_path, seed=0)
    assert (tmp_path / "results" / "E0-mean.json").exists()
    for mode in ("template", "image"):
        assert res["modes"][mode]["test"]["n"] == len(sets["test"])
        assert {"val", "test", "test_hard"} <= set(res["modes"][mode])
    # Children are colour mixes of their parents, so kin must score clearly higher.
    assert res["modes"]["template"]["test"]["roc_auc"] > 0.75


def test_e1_and_pairwise_run(eval_setup, tmp_path):
    out, store, sets = eval_setup
    tr, va = out["triplets"]["train"], out["triplets"]["val"]
    X_tr = handcrafted_features(*triplet_embeddings(store, tr, "template"))
    X_va = handcrafted_features(*triplet_embeddings(store, va, "template"))
    assert X_tr.shape == (len(tr), 8)
    scorer = ClassicalScorer("lr", [0.1, 1.0]).fit(X_tr, tr["label"], X_va, va["label"])
    assert scorer.chosen_c in (0.1, 1.0)
    res = evaluate_triplet_scorer("E1-LR", "test", scorer, store, sets, tmp_path, seed=0)
    assert 0 <= res["modes"]["template"]["test"]["accuracy"] <= 1
    pres = evaluate_pairwise_cosine("P0", "smoketest", store, {"val": out["pairs"]["val"], "test": out["pairs"]["test"]},
                                    tmp_path, seed=0)
    assert pres["modes"]["template"]["test"]["roc_auc"] > 0.6
