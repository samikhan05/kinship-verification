"""Train and evaluate the attention-gated tri-subject head on cached embeddings.

    E2          frozen ArcFace embeddings          (main model)
    E2-hardneg  same, but half of the training negatives are "one wrong parent"
    E2b         frozen FaceNet embeddings          (fair comparison for E3)

    python scripts/train_head.py --embeddings arcface --name E2
    python scripts/train_head.py --embeddings arcface --name E2-hardneg --set train.head.hard_negative_fraction=0.5
    python scripts/train_head.py --embeddings facenet --name E2b

A small hyper-parameter sweep (configs/default.yaml: train.head.sweep) is run and
the setting with the best VALIDATION ROC-AUC is kept; the test set plays no part.

Writes: <work_dir>/checkpoints/<name>.pt (+ a copy in <output_dir>/checkpoints/),
<output_dir>/logs/<name>_train.csv (per-epoch metrics of the chosen setting),
<output_dir>/logs/<name>_sweep.csv (validation AUC of every setting), <output_dir>/results/<name>.json,
<output_dir>/scores/<name>__*.csv (scores + attention weights alpha_F / alpha_M).

The JSON also holds an "alpha_analysis": on kin test triplets, does alpha_F go up
when the child is more similar to the father than to the mother? (Spearman
correlation with cos(F,C) - cos(M,C).) This checks whether the attention weights
can honestly be shown as "resembles father X% / mother Y%".
"""
import argparse
import itertools
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from kinship.embed import EmbeddingStore, load_embeddings  # noqa: E402
from kinship.evaluate import HEADLINE, evaluate_triplet_scorer, load_split_tables  # noqa: E402
from kinship.train import HeadScorer, alpha_analysis, load_head, train_head  # noqa: E402
from kinship.utils import add_common_args, config_from_args, get_logger, get_paths, seed_everything  # noqa: E402


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

    # Hyper-parameter sweep: train every combination, keep the best by VALIDATION AUC.
    grid = hp.get("sweep") or {}
    keys = sorted(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    runs = []
    for i, combo in enumerate(combos):
        hp_i = {k: v for k, v in hp.items() if k != "sweep"} | dict(zip(keys, combo))
        log.info(f"--- {name} config {i + 1}/{len(combos)}: {dict(zip(keys, combo))}")
        _, summ = train_head(store, t["triplets_train"], t["triplets_val"], persons, hp_i, seed, log,
                             output_dir / "logs" / f"{name}__cfg{i}_train.csv",
                             work_dir / "checkpoints" / f"{name}__cfg{i}.pt", t.get("hard_val"),
                             pairs_train=t.get("pairs_train"), pairs_val=t.get("pairs_val"))
        runs.append({"config": i, **dict(zip(keys, combo)), "best_val_auc": summ["best_val_auc"],
                     "best_epoch": summ["best_epoch"], "epochs_run": summ["epochs_run"]})
    sweep = pd.DataFrame(runs)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    sweep.to_csv(output_dir / "logs" / f"{name}_sweep.csv", index=False)
    best = int(sweep.loc[sweep["best_val_auc"].idxmax(), "config"])
    log.info(f"{name} sweep (validation AUC only):\n{sweep.to_string(index=False)}\n-> keeping config {best}")

    ckpt = work_dir / "checkpoints" / f"{name}.pt"
    shutil.copy2(work_dir / "checkpoints" / f"{name}__cfg{best}.pt", ckpt)
    shutil.copy2(output_dir / "logs" / f"{name}__cfg{best}_train.csv", output_dir / "logs" / f"{name}_train.csv")
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, output_dir / "checkpoints" / ckpt.name)
    model = load_head(ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    summary = {"chosen_config": best, "chosen_hyperparams": ck["hyperparams"], "best_epoch": ck["epoch"],
               "best_val_auc": ck["val_auc"], "model_args": ck["model_args"], "sweep": runs}

    device = next(model.parameters()).device.type
    scorer = HeadScorer(model, device)
    sets = {"val": t["triplets_val"], "test": t["triplets_test"]}
    if "hard_test" in t:
        sets["test_hard"] = t["hard_test"]
    backbone = {"arcface": "ArcFace", "facenet": "FaceNet"}.get(args.embeddings, args.embeddings)
    hn = float(hp["hard_negative_fraction"])
    desc = (f"attention-gated tri-subject head on frozen {backbone}"
            + (f", {hn:.0%} hard negatives in training" if hn > 0 else "")
            + (", comparator pretrained on pairs" if int(hp.get("pretrain_pair_epochs", 0)) > 0 else ""))
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
