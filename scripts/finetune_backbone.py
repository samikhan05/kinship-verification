"""E3: fine-tune FaceNet (last block) together with the tri-subject head.

    python scripts/finetune_backbone.py              # needs E2b first (its head is the start point)

Steps:
  1. Crop cache: all FaceNet 160x160 crops in one array
     (<work_dir>/embeddings/facenet/crops_160.npy), built once from the crops that
     extract_embeddings.py saved (here or in an attached earlier Kaggle run);
     missing crops are recomputed with MTCNN.
  2. Fine-tune (src/kinship/train.py: finetune_e3), early stopping on validation AUC.
  3. Embed every image with the fine-tuned backbone -> <work_dir>/embeddings/facenet_ft/
     (so `evaluate.py --embeddings facenet_ft` can also run E0/E1 on the new features).
  4. Evaluate exactly like E2/E2b -> <output_dir>/results/E3.json (+ attention analysis).

The InsightFace ArcFace model is an ONNX file and cannot be trained, which is why
fine-tuning uses FaceNet (a PyTorch model); E2b is the fair "before" for E3.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kinship.data import resolve_data_root  # noqa: E402
from kinship.embed import EmbeddingStore, build_crop_cache, load_embeddings  # noqa: E402
from kinship.evaluate import HEADLINE, evaluate_triplet_scorer, load_split_tables  # noqa: E402
from kinship.train import HeadScorer, alpha_analysis, embed_rows, finetune_e3, load_head  # noqa: E402
from kinship.utils import (add_common_args, config_from_args, find_in_inputs, get_logger,  # noqa: E402
                           get_paths, load_image_rgb, seed_everything)


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--name", default="E3")
    ap.add_argument("--embeddings", default="facenet", help="cached embeddings whose crops are fine-tuned on")
    args = ap.parse_args()
    cfg = config_from_args(args)
    _, work_dir, output_dir = get_paths(cfg)
    log = get_logger("finetune", work_dir / f"train_{args.name}.log")
    seed = int(cfg["seed"])
    seed_everything(seed)
    ft = cfg["finetune"]
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb_dir = work_dir / "embeddings" / args.embeddings
    emb, index = load_embeddings(emb_dir)
    store = EmbeddingStore(emb, index)
    t = load_split_tables(work_dir / "splits", store, log)
    persons = pd.read_csv(work_dir / "splits" / "persons.csv")

    # 1. crop cache
    size = int(cfg["facenet"]["image_size"])
    crop_dirs = [work_dir / "aligned" / args.embeddings] + find_in_inputs("work", "aligned", args.embeddings)
    crop_dirs = [d for d in crop_dirs if d.is_dir()]
    log.info(f"Looking for saved crops in: {[str(d) for d in crop_dirs] or 'nowhere - will recompute'}")
    aligner = {}

    def recompute(rel: str) -> np.ndarray:
        """Missing crop: run MTCNN again (slow, only used for missing files)."""
        if "a" not in aligner:
            from kinship.align import make_aligner

            aligner["a"] = make_aligner(cfg, args.embeddings, log)
        root = resolve_data_root(cfg["paths"]["data_root"])
        return aligner["a"].process(load_image_rgb(root / rel)).crop

    crops = build_crop_cache(index, emb_dir / f"crops_{size}.npy", size, crop_dirs, recompute, log)

    # 2. fine-tune, starting from the E2b head
    init = work_dir / "checkpoints" / f"{ft['init_head']}.pt"
    if not init.exists():
        sys.exit(f"{init} not found: run scripts/train_head.py --embeddings {args.embeddings} "
                 f"--name {ft['init_head']} first.")
    head = load_head(init, "cpu")
    log.info(f"{args.name}: fine-tuning FaceNet blocks {ft['unfreeze']} + head from {init.name} on {device}")
    ckpt = work_dir / "checkpoints" / f"{args.name}.pt"
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    model, summary = finetune_e3(store, index, crops, head, t["triplets_train"], t["triplets_val"], persons, ft,
                                 seed, log, output_dir / "logs" / f"{args.name}_train.csv", ckpt,
                                 t.get("hard_val"), pretrained=cfg["facenet"]["pretrained"], device=device)

    # 3. embeddings of every image with the fine-tuned backbone
    ft_dir = work_dir / "embeddings" / f"{args.embeddings}_ft"
    ft_dir.mkdir(parents=True, exist_ok=True)
    emb_ft = embed_rows(model, crops, np.arange(len(index)), device)
    np.save(ft_dir / "embeddings.npy", emb_ft)
    index.to_csv(ft_dir / "index.csv", index=False)
    log.info(f"Fine-tuned embeddings {emb_ft.shape} -> {ft_dir}")

    # 4. evaluate exactly like E2 / E2b
    store_ft = EmbeddingStore(emb_ft, index)
    scorer = HeadScorer(model.head, device)
    sets = {"val": t["triplets_val"], "test": t["triplets_test"]}
    if "hard_test" in t:
        sets["test_hard"] = t["hard_test"]
    desc = (f"FaceNet fine-tuned end to end ({', '.join(ft['unfreeze'])} unfrozen) "
            f"with the tri-subject head, started from {ft['init_head']}")
    alpha = alpha_analysis(store_ft, t["triplets_test"], scorer)
    res = evaluate_triplet_scorer(args.name, desc, scorer, store_ft, sets, output_dir, seed,
                                  float(cfg["eval"]["far"]),
                                  extra={"training": summary, "alpha_analysis": alpha, "embeddings": f"{args.embeddings}_ft"})
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, output_dir / "checkpoints" / ckpt.name)
    a, x = res["modes"]["template"]["test"], res["modes"][HEADLINE]["test"]
    log.info(f"{args.name}: test acc / auc - all photos {a['accuracy']:.3f} / {a['roc_auc']:.3f}, "
             f"different photos {x['accuracy']:.3f} / {x['roc_auc']:.3f} (n={x['n']}); "
             f"hard (different photos) {res['modes'][HEADLINE].get('test_hard', {}).get('roc_auc', float('nan')):.3f} AUC")
    log.info(f"{args.name}: attention analysis {alpha}")


if __name__ == "__main__":
    main()
