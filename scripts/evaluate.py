"""Run the baseline experiments on cached embeddings and save results JSON + scores.

    E0  frozen ArcFace + cosine, no training:
          E0-mean = mean(cos(F,C), cos(M,C)),  E0-max = max(...)
          P0-cosine = pairwise parent-child baseline cos(p1, p2)
    E1  classical ML on hand-made features from frozen ArcFace embeddings:
          E1-LR (logistic regression), E1-SVM (linear SVM)

Outputs: <output_dir>/results/<exp>.json and <output_dir>/scores/*.csv.
Then run scripts/make_report.py to build results.md.

    python scripts/evaluate.py --exp E0 E1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kinship.embed import EmbeddingStore, load_embeddings  # noqa: E402
from kinship.evaluate import (HEADLINE, ClassicalScorer, CosineScorer, load_split_tables,  # noqa: E402
                              evaluate_pairwise_cosine, evaluate_triplet_scorer, handcrafted_features,
                              keep_complete_pairs, triplet_embeddings)
from kinship.utils import add_common_args, config_from_args, get_logger, get_paths, seed_everything  # noqa: E402


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--exp", nargs="+", default=["E0", "E1"], choices=["E0", "E1"])
    ap.add_argument("--embeddings", default="arcface", help="which cached embeddings to use")
    args = ap.parse_args()
    cfg = config_from_args(args)
    _, work_dir, output_dir = get_paths(cfg)
    log = get_logger("evaluate", work_dir / "evaluate.log")
    seed = int(cfg["seed"])
    seed_everything(seed)
    far = float(cfg["eval"]["far"])

    store = EmbeddingStore(*load_embeddings(work_dir / "embeddings" / args.embeddings))
    t = load_split_tables(work_dir / "splits", store, log)
    sets = {"val": t["triplets_val"], "test": t["triplets_test"]}
    if "hard_test" in t:
        sets["test_hard"] = t["hard_test"]
    tag = "" if args.embeddings == "arcface" else f" ({args.embeddings})"
    backbone = {"arcface": "ArcFace", "facenet": "FaceNet"}.get(args.embeddings, args.embeddings)

    def show(res):
        a, x = res["modes"]["template"]["test"], res["modes"][HEADLINE]["test"]
        log.info(f"{res['experiment']}: test acc / auc - all photos {a['accuracy']:.3f} / {a['roc_auc']:.3f}, "
                 f"different photos {x['accuracy']:.3f} / {x['roc_auc']:.3f} (n={x['n']})")

    if "E0" in args.exp:
        for how in ("mean", "max"):
            show(evaluate_triplet_scorer(
                f"E0-{how}{tag}", f"frozen {backbone}, {how} of cos(F,C) and cos(M,C), no training",
                CosineScorer(how), store, sets, output_dir, seed, far))
        show(evaluate_pairwise_cosine(f"P0-cosine{tag}", backbone, store,
                                      {"val": t["pairs_val"], "test": t["pairs_test"]}, output_dir, seed, far))

    if "E1" in args.exp:
        # Features from per-person templates built from DIFFERENT photos, so the model
        # cannot learn the same-photo shortcut. Scaler + C: fitted on train / chosen on val only.
        def feats(df):
            *emb, ok = triplet_embeddings(store, df, "template", cross_photo=True)
            keep = keep_complete_pairs(df, ok)
            return handcrafted_features(*(e[keep] for e in emb)), df["label"].to_numpy()[keep]
        (X_tr, y_tr), (X_va, y_va) = feats(t["triplets_train"]), feats(t["triplets_val"])
        for kind, label in (("lr", "logistic regression"), ("svm", "linear SVM")):
            scorer = ClassicalScorer(kind, cfg["eval"]["e1_c_grid"], seed).fit(X_tr, y_tr, X_va, y_va)
            show(evaluate_triplet_scorer(
                f"E1-{kind.upper()}{tag}", f"{label} on 8 hand-made features (frozen {backbone})",
                scorer, store, sets, output_dir, seed, far,
                extra={"chosen_C": scorer.chosen_c, "val_auc_by_C": scorer.val_auc_by_c,
                       "train_triplets": int(len(y_tr))}))
    log.info(f"Results in {output_dir / 'results'}. Next: python scripts/make_report.py")


if __name__ == "__main__":
    main()
