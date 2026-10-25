"""Train and evaluate the attention-gated tri-subject head on cached embeddings.

    E2          frozen ArcFace embeddings          (main model)
    E2-hardneg  same, but half of the training negatives are "one wrong parent"
    E2b         frozen FaceNet embeddings          (fair comparison for E3)

    python scripts/train_head.py --embeddings arcface --name E2
    python scripts/train_head.py --embeddings arcface --name E2-hardneg --set train.head.hard_negative_fraction=0.5
    python scripts/train_head.py --embeddings facenet --name E2b

Writes: <work_dir>/checkpoints/<name>.pt (+ a copy in <output_dir>/checkpoints/),
<output_dir>/logs/<name>_train.csv (per-epoch metrics), <output_dir>/results/<name>.json,
<output_dir>/scores/<name>__*.csv (scores + attention weights alpha_F / alpha_M).

The JSON also holds an "alpha_analysis": on kin test triplets, does alpha_F go up
when the child is more similar to the father than to the mother? (Spearman
correlation with cos(F,C) - cos(M,C).) This checks whether the attention weights
can honestly be shown as "resembles father X% / mother Y%".
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from kinship.embed import EmbeddingStore, load_embeddings  # noqa: E402
from kinship.evaluate import (HEADLINE, evaluate_triplet_scorer,  # noqa: E402
                              load_split_tables, rowwise_cos, triplet_embeddings)
from kinship.train import HeadScorer, train_head  # noqa: E402
from kinship.utils import add_common_args, config_from_args, get_logger, get_paths, seed_everything  # noqa: E402


def alpha_analysis(store: EmbeddingStore, test: pd.DataFrame, scorer: HeadScorer) -> dict:
    """How the attention weights behave on KIN test triplets (headline protocol)."""
    pos = test[test["label"] == 1].reset_index(drop=True)
    *emb, ok = triplet_embeddings(store, pos, "template", cross_photo=True)
    eF, eM, eC = (e[ok] for e in emb)
    pos = pos[ok].reset_index(drop=True)
    _, d = scorer.score_details(eF, eM, eC)
    cos_gap = rowwise_cos(eF, eC) - rowwise_cos(eM, eC)          # > 0: child closer to father
    rho, pval = spearmanr(d["alpha_F"], cos_gap)
    agree = float(np.mean((d["alpha_F"] > 0.5) == (cos_gap > 0)))
    out = {"n_kin_triplets": int(len(pos)),
           "spearman_alphaF_vs_cos_gap": float(rho), "spearman_p_value": float(pval),
           "share_where_alpha_picks_more_similar_parent": agree,
           "mean_alpha_F": float(d["alpha_F"].mean()),
           "share_alpha_F_above_half": float((d["alpha_F"] > 0.5).mean())}
    for kind in ("FMS", "FMD"):
        m = (pos["kind"] == kind).to_numpy()
        if m.any():
            out[f"mean_alpha_F_{kind}"] = float(d["alpha_F"][m].mean())
    return out


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--embeddings", default="arcface", choices=["arcface", "facenet", "smoketest"])
    ap.add_argument("--name", default=None, help="experiment name (default: E2 for arcface, E2b for facenet)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    _, work_dir, output_dir = get_paths(cfg)
    name = args.name or {"arcface": "E2", "facenet": "E2b"}.get(args.embeddings, f"E2-{args.embeddings}")
    log = get_logger("train_head", work_dir / f"train_{name}.log")
    seed = int(cfg["seed"])
    seed_everything(seed)
    hp = cfg["train"]["head"]

    store = EmbeddingStore(*load_embeddings(work_dir / "embeddings" / args.embeddings))
    t = load_split_tables(work_dir / "splits", store, log)
    persons = pd.read_csv(work_dir / "splits" / "persons.csv")
    log.info(f"{name}: training on {args.embeddings} embeddings, {int((t['triplets_train'].label == 1).sum())} "
             f"kin training triplets, hard_negative_fraction={hp['hard_negative_fraction']}")

    ckpt = work_dir / "checkpoints" / f"{name}.pt"
    model, summary = train_head(store, t["triplets_train"], t["triplets_val"], persons, hp, seed, log,
                                output_dir / "logs" / f"{name}_train.csv", ckpt, t.get("hard_val"))
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, output_dir / "checkpoints" / ckpt.name)

    device = next(model.parameters()).device.type
    scorer = HeadScorer(model, device)
    sets = {"val": t["triplets_val"], "test": t["triplets_test"]}
    if "hard_test" in t:
        sets["test_hard"] = t["hard_test"]
    backbone = {"arcface": "ArcFace", "facenet": "FaceNet"}.get(args.embeddings, args.embeddings)
    hn = float(hp["hard_negative_fraction"])
    desc = (f"attention-gated tri-subject head on frozen {backbone}"
            + (f", {hn:.0%} hard negatives in training" if hn > 0 else ""))
    alpha = alpha_analysis(store, t["triplets_test"], scorer)
    res = evaluate_triplet_scorer(name, desc, scorer, store, sets, output_dir, seed,
                                  float(cfg["eval"]["far"]),
                                  extra={"training": summary, "alpha_analysis": alpha,
                                         "embeddings": args.embeddings})
    a, x = res["modes"]["template"]["test"], res["modes"][HEADLINE]["test"]
    log.info(f"{name}: test acc / auc - all photos {a['accuracy']:.3f} / {a['roc_auc']:.3f}, "
             f"different photos {x['accuracy']:.3f} / {x['roc_auc']:.3f} (n={x['n']}); "
             f"hard (different photos) {res['modes'][HEADLINE].get('test_hard', {}).get('roc_auc', float('nan')):.3f} AUC")
    log.info(f"{name}: attention analysis {alpha}")


if __name__ == "__main__":
    main()
