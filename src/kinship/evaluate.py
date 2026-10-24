"""Metrics, threshold selection and the evaluation protocol shared by all experiments.

Protocol (the same for E0 ... E4, so the rows of results.md are comparable):
  * The decision threshold is chosen on the VALIDATION set only (max accuracy),
    then applied unchanged to the test sets.
  * Metrics: accuracy, ROC-AUC, F1, TAR at FAR = 0.1.
  * Breakdowns: FMS vs FMD, main test set vs hard-negative set, and two scoring
    modes - "template" (mean embedding over all images of a person) and "image"
    (one randomly chosen image per person, fixed seed).
  * Every score is saved to outputs/scores/ so plots can be re-made later.

An experiment is any object with `score(eF, eM, eC) -> scores` (higher = more
likely kin) and optionally `fit(...)`. Scorers are defined here for E0 and E1;
the learned heads (E2, E3) plug into the same functions later.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from .embed import EmbeddingStore
from .utils import ensure_dir, save_json

MODES = ("template", "image")


# ---------------------------------------------------------------------- metrics
def choose_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold t that maximises accuracy of the rule "kin if score >= t".

    Sort scores from high to low; predicting the top k as kin gives
    accuracy (TP_k + TN_k) / n. Try every k at a boundary between distinct
    scores and return the midpoint between the k-th and (k+1)-th score.
    """
    y, s = np.asarray(y, int), np.asarray(s, float)
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    n, n_pos = len(y), int(y.sum())
    tp = np.concatenate([[0], np.cumsum(y_sorted)])           # kin among the top k
    fp = np.concatenate([[0], np.cumsum(1 - y_sorted)])
    acc = (tp + (n - n_pos - fp)) / n                          # k = 0 .. n
    valid = np.ones(n + 1, bool)
    valid[1:n] = s_sorted[:-1] > s_sorted[1:]                  # can't split between equal scores
    k = int(np.flatnonzero(valid)[np.argmax(acc[valid])])
    if k == 0:
        return float(s_sorted[0] + 1e-6)
    if k == n:
        return float(s_sorted[-1] - 1e-6)
    return float((s_sorted[k - 1] + s_sorted[k]) / 2)


def tar_at_far(y: np.ndarray, s: np.ndarray, far: float = 0.1) -> float:
    """True accept rate when at most `far` of the non-kin are accepted."""
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= far + 1e-12].max())


def compute_metrics(y, s, threshold: float, far: float = 0.1) -> dict:
    y, s = np.asarray(y, int), np.asarray(s, float)
    pred = (s >= threshold).astype(int)
    both = 0 < y.sum() < len(y)
    return {"n": int(len(y)), "n_pos": int(y.sum()),
            "accuracy": float((pred == y).mean()),
            "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
            "f1": float(f1_score(y, pred, zero_division=0)),
            f"tar@far={far}": tar_at_far(y, s, far) if both else float("nan"),
            "threshold": float(threshold)}


# ---------------------------------------------------------------------- features
def rowwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity per row (inputs are already L2-normalised)."""
    return np.sum(a * b, axis=1)


def triplet_embeddings(store: EmbeddingStore, df: pd.DataFrame, mode: str, rng=None):
    """(eF, eM, eC) for each row of a triplet table."""
    return tuple(store.stack(df[col].tolist(), mode, rng) for col in ("F", "M", "C"))


def handcrafted_features(eF, eM, eC) -> np.ndarray:
    """E1 features: [cos_FC, cos_MC, mean|eF-eC|, mean|eM-eC|, and mean / std / max /
    min of the element-wise product |eF-eC| * |eM-eC|]. The product is large in the
    dimensions where the child differs from BOTH parents."""
    dF, dM = np.abs(eF - eC), np.abs(eM - eC)
    prod = dF * dM
    return np.column_stack([rowwise_cos(eF, eC), rowwise_cos(eM, eC), dF.mean(1), dM.mean(1),
                            prod.mean(1), prod.std(1), prod.max(1), prod.min(1)])


# ---------------------------------------------------------------------- scorers
class CosineScorer:
    """E0: no training. Score = mean (or max) of cos(F, C) and cos(M, C)."""

    def __init__(self, how: str = "mean"):
        self.how = how

    def score(self, eF, eM, eC) -> np.ndarray:
        both = np.column_stack([rowwise_cos(eF, eC), rowwise_cos(eM, eC)])
        return both.mean(1) if self.how == "mean" else both.max(1)


class ClassicalScorer:
    """E1: logistic regression or linear SVM on `handcrafted_features`.

    The scaler is fitted on training data only (it lives inside the pipeline) and
    the regularisation strength C is picked by validation ROC-AUC.
    """

    def __init__(self, kind: str = "lr", c_grid=(0.01, 0.1, 1.0, 10.0), seed: int = 0):
        self.kind, self.c_grid, self.seed = kind, list(c_grid), seed
        self.model, self.chosen_c, self.val_auc_by_c = None, None, {}

    def _make(self, c: float):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        clf = (LogisticRegression(C=c, max_iter=2000) if self.kind == "lr"
               else LinearSVC(C=c, max_iter=20000, random_state=self.seed))
        return make_pipeline(StandardScaler(), clf)

    def fit(self, X_train, y_train, X_val, y_val) -> "ClassicalScorer":
        best = -1.0
        for c in self.c_grid:
            m = self._make(c).fit(X_train, y_train)
            auc = roc_auc_score(y_val, m.decision_function(X_val))
            self.val_auc_by_c[str(c)] = float(auc)
            if auc > best:
                best, self.model, self.chosen_c = auc, m, c
        return self

    def score(self, eF, eM, eC) -> np.ndarray:
        return self.model.decision_function(handcrafted_features(eF, eM, eC))


# ---------------------------------------------------------------------- running
def evaluate_triplet_scorer(name: str, description: str, scorer, store: EmbeddingStore,
                            sets: dict[str, pd.DataFrame], out_dir: str | Path, seed: int,
                            far: float = 0.1, extra: dict | None = None) -> dict:
    """Score val / test / test_hard in both modes, pick the threshold on val, save
    results/<name>.json and scores/<name>__<set>__<mode>.csv.

    `sets` must contain 'val' and 'test' (main sets) and may contain 'test_hard'.
    """
    out_dir = Path(out_dir)
    score_dir = ensure_dir(out_dir / "scores")
    result = {"experiment": name, "description": description, "protocol":
              "threshold = max accuracy on val (main set, same scoring mode); applied unchanged to test",
              "modes": {}, **(extra or {})}
    for m_i, mode in enumerate(MODES):
        scored = {}
        for s_i, (set_name, df) in enumerate(sets.items()):
            rng = np.random.default_rng([seed, m_i, s_i])     # fixed image choice per set
            s = scorer.score(*triplet_embeddings(store, df, mode, rng))
            scored[set_name] = s
            df.assign(score=s, mode=mode).to_csv(score_dir / f"{name}__{set_name}__{mode}.csv", index=False)
        thr = choose_threshold(sets["val"]["label"].to_numpy(), scored["val"])
        res = {}
        for set_name, df in sets.items():
            res[set_name] = compute_metrics(df["label"], scored[set_name], thr, far)
        test = sets["test"]
        for kind in sorted(test["kind"].unique()):
            mask = (test["kind"] == kind).to_numpy()
            res[f"test_{kind}"] = compute_metrics(test["label"][mask], scored["test"][mask], thr, far)
        result["modes"][mode] = res
    save_json(result, ensure_dir(out_dir / "results") / f"{name}.json")
    return result


def evaluate_pairwise_cosine(name: str, backbone: str, store: EmbeddingStore, pairs: dict[str, pd.DataFrame],
                             out_dir: str | Path, seed: int, far: float = 0.1) -> dict:
    """Pairwise parent-child baseline: score = cos(p1, p2). Breakdown by relation
    (FS / FD / MS / MD when roles are known, 'related' otherwise)."""
    out_dir = Path(out_dir)
    result = {"experiment": name, "description": f"frozen {backbone}, cosine similarity of the pair",
              "task": "pairwise", "modes": {}}
    for m_i, mode in enumerate(MODES):
        scored = {}
        for s_i, (set_name, df) in enumerate(pairs.items()):
            rng = np.random.default_rng([seed, 10 + m_i, s_i])
            a = store.stack(df["p1"].tolist(), mode, rng)
            b = store.stack(df["p2"].tolist(), mode, rng)
            scored[set_name] = rowwise_cos(a, b)
            df.assign(score=scored[set_name], mode=mode).to_csv(
                ensure_dir(out_dir / "scores") / f"{name}__{set_name}__{mode}.csv", index=False)
        thr = choose_threshold(pairs["val"]["label"].to_numpy(), scored["val"])
        res = {s: compute_metrics(df["label"], scored[s], thr, far) for s, df in pairs.items()}
        test = pairs["test"]
        for rel in sorted(test["rel"].unique()):
            mask = (test["rel"] == rel).to_numpy()
            res[f"test_{rel}"] = compute_metrics(test["label"][mask], scored["test"][mask], thr, far)
        result["modes"][mode] = res
    save_json(result, ensure_dir(out_dir / "results") / f"{name}.json")
    return result
