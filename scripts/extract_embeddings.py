"""Detect + align every face, compute ArcFace embeddings and gender/age (cached).

Writes to <work_dir>/:
    embeddings/<aligner>/chunk_*.{csv,npy}   resumable partial results
    embeddings/<aligner>/embeddings.npy      N x 512, L2-normalised
    embeddings/<aligner>/index.csv           one row per image (person, detected?, gender, age, ...)
    embeddings/<aligner>/detection.json      detection rate, fallbacks, multi-face count
    aligned/<aligner>/...                    aligned 112x112 crops (jpg), same tree as the dataset
    attributes.csv                           one estimated gender + age per person (prepare.py needs it)

Re-running continues from the last finished chunk.

    python scripts/extract_embeddings.py                         # Kaggle, InsightFace on GPU
    python scripts/extract_embeddings.py --aligner smoketest ...  # CPU smoke test, no InsightFace
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kinship.align import make_aligner  # noqa: E402
from kinship.data import load_dataset  # noqa: E402
from kinship.embed import detection_report, extract_embeddings, person_attributes  # noqa: E402
from kinship.utils import (add_common_args, config_from_args, get_logger, get_paths,  # noqa: E402
                           save_json, seed_everything)


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--aligner", choices=["arcface", "smoketest"], default="arcface")
    ap.add_argument("--limit", type=int, help="only the first N images (quick test)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    _, work_dir, output_dir = get_paths(cfg)
    log = get_logger("extract", work_dir / "extract.log")
    seed_everything(int(cfg["seed"]))

    ds = load_dataset(cfg)
    log.info(f"Loaded {ds.format} dataset: {len(ds.persons)} persons, {ds.n_images} images")
    aligner = make_aligner(cfg, args.aligner, log)
    log.info(f"Aligner: {aligner.name} on {aligner.device}")

    emb_dir = work_dir / "embeddings" / aligner.name
    crops_dir = work_dir / "aligned" / aligner.name if cfg["align"].get("save_crops", True) else None
    emb, index = extract_embeddings(ds, aligner, emb_dir, int(cfg["embed"]["chunk_size"]),
                                    crops_dir, args.limit, log)
    log.info(f"Embeddings: {emb.shape} -> {emb_dir}")

    report = detection_report(index)
    save_json(report, emb_dir / "detection.json")
    save_json(report, output_dir / f"detection_{aligner.name}.json")
    log.info(f"Detection: {report}")

    attrs = person_attributes(index)
    if len(attrs):
        attrs.to_csv(work_dir / "attributes.csv", index=False)
        log.info(f"Estimated gender/age for {len(attrs)} of {len(ds.persons)} persons -> attributes.csv")
        # Sanity check against FIW ground-truth gender, where available.
        truth = {p.pid: p.gender for p in ds.persons.values() if p.gender}
        both = attrs[attrs["pid"].isin(truth)]
        if len(both):
            agree = (both["gender"] == both["pid"].map(truth)).mean()
            log.info(f"Estimated gender agrees with FIW labels for {agree:.1%} of {len(both)} persons")


if __name__ == "__main__":
    main()
