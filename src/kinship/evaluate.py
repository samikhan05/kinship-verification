"""Metrics, threshold selection and the evaluation protocol shared by all experiments.

Protocol (the same for E0 ... E4, so the rows of results.md are comparable):
  * The decision threshold is chosen on the VALIDATION set only (max accuracy),
    then applied unchanged to the test sets.
  * Metrics: accuracy, ROC-AUC, F1, TAR at FAR = 0.1.
  * Four scoring protocols:
      template         mean embedding over all images of each person
      image            one randomly chosen image per person (fixed seed)
      template_xphoto  like template, but the child's images come from photos
      image_xphoto     that neither parent's images come from
    Why "xphoto": in FIW, relatives are often cut from the same photo (94% of kin
    test triplets share a photo between child and a parent; non-kin never do).
    Same-photo faces share lighting/camera/colour, which a model can exploit
    without learning kinship. The xphoto protocols remove that shortcut and are
    the headline numbers. A kin triplet that cannot be scored this way is
    dropped together with its matched non-kin triplet, so sets stay 50/50.
  * Breakdowns: FMS vs FMD, main test set vs hard-negative set.
  * Every score is saved to outputs/scores/ so plots can be re-made later.

An experiment is any object with `score(eF, eM, eC) -> scores` (higher = more
likely kin). A scorer may also define `score_details(eF, eM, eC) -> (scores,
{column: values})`; the extra columns (e.g. E2's attention weights) are saved
next to the scores.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from .embed import EmbeddingStore
from .utils import ensure_dir, save_json

# protocol name -> (embedding mode, compare only images from different photos?)
PROTOCOLS = {"template": ("template", False), "image": ("image", False),
             "template_xphoto": ("template", True), "image_xphoto": ("image", True)}
HEADLINE = "template_xphoto"


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


# ---------------------------------------------------------------------- data helpers
def pair_ids(df: pd.DataFrame) -> np.ndarray | None:
    """Which kin row each non-kin row was made from (they share a pair_id).

    Older split files have no pair_id column; their triplet tables list the kin
    rows first and the negatives in the same order, so position gives the pairing.
    """
    if "pair_id" in df:
        return df["pair_id"].to_numpy()
    n, lab = len(df) // 2, df["label"].to_numpy()
    if len(df) % 2 == 0 and (lab[:n] == 1).all() and (lab[n:] == 0).all():
        return np.concatenate([np.arange(n), np.arange(n)])
    return None


def keep_complete_pairs(df: pd.DataFrame, ok: np.ndarray) -> np.ndarray:
    """Drop rows that could not be scored AND their matched partner (keeps 50/50)."""
    ids = pair_ids(df)
    if ids is None:
        return ok
    return ok & ~np.isin(ids, ids[~ok])


def load_split_tables(split_dir: str | Path, store: EmbeddingStore, log=None) -> dict[str, pd.DataFrame]:
    """Read the prepared split CSVs. Rows with a person that has no embedding are
    dropped together with their matched partner row (keeps sets 50/50)."""
    names = ["triplets_train", "triplets_val", "triplets_test", "hard_val", "hard_test",
             "pairs_train", "pairs_val", "pairs_test"]
    tables = {}
    for name in names:
        path = Path(split_dir) / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        cols = [c for c in ("F", "M", "C", "p1", "p2") if c in df]
        ok = df[cols].apply(lambda col: col.map(lambda p: p in store)).all(axis=1).to_numpy()
        keep = keep_complete_pairs(df, ok)
        if (~keep).any() and log is not None:
            log.warning(f"{name}: dropping {(~keep).sum()} rows (person without embedding, or its partner)")
        tables[name] = df[keep].reset_index(drop=True)
    return tables


def rowwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity per row (inputs are already L2-normalised)."""
    return np.sum(a * b, axis=1)


def triplet_embeddings(store: EmbeddingStore, df: pd.DataFrame, mode: str, rng=None,
                       cross_photo: bool = False):
    """(eF, eM, eC, ok) for each row of a triplet table. With cross_photo, the
    child's images come from photos that neither parent's images come from
    (`EmbeddingStore.triplet_rows`); rows where that is impossible have ok=False."""
    return store.triplet(df["F"], df["M"], df["C"], mode, rng, cross_photo)


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
    """Score val / test / test_hard under every protocol, pick the threshold on val,
    save results/<name>.json and scores/<name>__<set>__<protocol>.csv.

    `sets` must contain 'val' and 'test' (main sets) and may contain 'test_hard'.
    """
    out_dir = Path(out_dir)
    score_dir = ensure_dir(out_dir / "scores")
    result = {"experiment": name, "description": description, "headline_protocol": HEADLINE,
              "protocol": "threshold = max accuracy on val (main set, same protocol); applied unchanged to test",
              "modes": {}, **(extra or {})}
    for p_i, (proto, (mode, xphoto)) in enumerate(PROTOCOLS.items()):
        scored, kept = {}, {}
        for s_i, (set_name, df) in enumerate(sets.items()):
            rng = np.random.default_rng([seed, p_i, s_i])     # fixed image choice per set
            *emb, ok = triplet_embeddings(store, df, mode, rng, xphoto)
            keep = keep_complete_pairs(df, ok)
            kept[set_name] = df[keep].reset_index(drop=True)
            emb = [e[keep] for e in emb]
            if hasattr(scorer, "score_details"):
                scored[set_name], details = scorer.score_details(*emb)
            else:
                scored[set_name], details = scorer.score(*emb), {}
            kept[set_name].assign(score=scored[set_name], protocol=proto, **details).to_csv(
                score_dir / f"{name}__{set_name}__{proto}.csv", index=False)
        thr = choose_threshold(kept["val"]["label"].to_numpy(), scored["val"])
        res = {}
        for set_name, df in kept.items():
            res[set_name] = compute_metrics(df["label"], scored[set_name], thr, far)
            res[set_name]["dropped_rows"] = int(len(sets[set_name]) - len(df))
        test = kept["test"]
        for kind in sorted(test["kind"].unique()):
            mask = (test["kind"] == kind).to_numpy()
            res[f"test_{kind}"] = compute_metrics(test["label"][mask], scored["test"][mask], thr, far)
        result["modes"][proto] = res
    save_json(result, ensure_dir(out_dir / "results") / f"{name}.json")
    return result


def evaluate_pairwise_cosine(name: str, backbone: str, store: EmbeddingStore,
                             pairs: dict[str, pd.DataFrame], out_dir: str | Path, seed: int,
                             far: float = 0.1) -> dict:
    """Pairwise parent-child baseline: score = cos(p1, p2). Breakdown by relation
    (FS / FD / MS / MD when roles are known, 'related' otherwise)."""
    out_dir = Path(out_dir)
    result = {"experiment": name, "description": f"frozen {backbone}, cosine similarity of the pair",
              "task": "pairwise", "headline_protocol": HEADLINE, "modes": {}}
    for p_i, (proto, (mode, xphoto)) in enumerate(PROTOCOLS.items()):
        scored, kept = {}, {}
        for s_i, (set_name, df) in enumerate(pairs.items()):
            rng = np.random.default_rng([seed, 10 + p_i, s_i])
            a, b, ok = store.pair(df["p1"], df["p2"], mode, rng, xphoto)
            keep = keep_complete_pairs(df, ok)
            kept[set_name] = df[keep].reset_index(drop=True)
            scored[set_name] = rowwise_cos(a[keep], b[keep])
            kept[set_name].assign(score=scored[set_name], protocol=proto).to_csv(
                ensure_dir(out_dir / "scores") / f"{name}__{set_name}__{proto}.csv", index=False)
        thr = choose_threshold(kept["val"]["label"].to_numpy(), scored["val"])
        res = {s: compute_metrics(df["label"], scored[s], thr, far) for s, df in kept.items()}
        test = kept["test"]
        for rel in sorted(test["rel"].unique()):
            mask = (test["rel"] == rel).to_numpy()
            res[f"test_{rel}"] = compute_metrics(test["label"][mask], scored["test"][mask], thr, far)
        result["modes"][proto] = res
    save_json(result, ensure_dir(out_dir / "results") / f"{name}.json")
    return result
