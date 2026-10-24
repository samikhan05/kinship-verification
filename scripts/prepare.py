"""Build family-disjoint splits, positive triplets, negatives and pairwise pairs.

Writes to <work_dir>/splits/:
    families_{train,val,test}.txt   family ids per split
    triplets_{split}.csv            balanced main set: (F, M, C) vs (F, M, C')
    hard_{val,test}.csv             balanced hard set: (F, M, C) vs one wrong parent
    pairs_{split}.csv               parent-child / related pairs vs strangers
    persons.csv                     every person with gender, age, #images, split
    summary.json                    counts (incl. how many pseudo-triplets were mined)
and <output_dir>/triplet_check.jpg (30 random positive triplets F | M | C).

Option B needs estimated gender/age per person: by default it reads
<work_dir>/attributes.csv, which extract_embeddings.py writes. Use --attributes to
point elsewhere (e.g. the ground-truth file of the synthetic data).

Cached: if summary.json already exists, nothing is rebuilt unless --force.

    python scripts/prepare.py --data_root data/synth_kaggle --work_dir work --output_dir outputs \
        --attributes data/synth_kaggle/attributes_gt.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kinship.data import apply_attributes, load_dataset  # noqa: E402
from kinship.triplets import SPLITS, build_all_splits  # noqa: E402
from kinship.utils import (add_common_args, config_from_args, ensure_dir, get_logger,  # noqa: E402
                           get_paths, make_contact_sheet, save_json, seed_everything)


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--attributes", help="CSV with pid,gender,age (default: <work_dir>/attributes.csv)")
    ap.add_argument("--force", action="store_true", help="rebuild even if splits already exist")
    args = ap.parse_args()
    cfg = config_from_args(args)
    _, work_dir, output_dir = get_paths(cfg)
    split_dir = ensure_dir(work_dir / "splits")
    log = get_logger("prepare", work_dir / "prepare.log")
    seed_everything(int(cfg["seed"]))

    if (split_dir / "summary.json").exists() and not args.force:
        log.info(f"Splits already exist in {split_dir} (use --force to rebuild). Nothing to do.")
        return

    ds = load_dataset(cfg)
    log.info(f"Loaded {ds.format} dataset: {len(ds.families)} families, {len(ds.persons)} persons")

    attr_path = Path(args.attributes) if args.attributes else work_dir / "attributes.csv"
    attrs = pd.read_csv(attr_path) if attr_path.exists() else None
    if attrs is None and ds.format == "kaggle":
        sys.exit(f"Option B needs estimated gender/age per person, but {attr_path} does not exist. "
                 "Run scripts/extract_embeddings.py first (or pass --attributes).")
    n_attr = apply_attributes(ds, attrs)
    log.info(f"Attributes applied to {n_attr} persons" + (f" from {attr_path}" if attrs is not None else ""))

    result = build_all_splits(ds, cfg)

    for split in SPLITS:
        (split_dir / f"families_{split}.txt").write_text("\n".join(result["families"][split]) + "\n")
        result["triplets"][split].to_csv(split_dir / f"triplets_{split}.csv", index=False)
        result["pairs"][split].to_csv(split_dir / f"pairs_{split}.csv", index=False)
        if split in result["hard"]:
            result["hard"][split].to_csv(split_dir / f"hard_{split}.csv", index=False)

    fid_to_split = {fid: s for s in SPLITS for fid in result["families"][s]}
    pd.DataFrame([{"pid": p.pid, "fid": p.fid, "gender": p.gender, "age": p.age,
                   "n_images": len(p.images), "split": fid_to_split[p.fid]}
                  for p in ds.persons.values()]).to_csv(split_dir / "persons.csv", index=False)
    save_json(result["stats"], split_dir / "summary.json")

    pos = result["stats"]["positives"]
    log.info(f"Positive triplets ({pos['source']}): {pos}")
    for split, s in result["stats"]["splits"].items():
        log.info(f"{split:5s}: {s}")

    # Contact sheet of random positives for checking the triplets by eye.
    all_pos = pd.concat([result["triplets"][s].query("label == 1") for s in SPLITS])
    if len(all_pos):
        n = min(int(cfg["viz"]["contact_sheet_rows"]), len(all_pos))
        rng = np.random.default_rng(int(cfg["seed"]))
        picks = all_pos.iloc[rng.choice(len(all_pos), n, replace=False)]
        rows, captions = [], []
        for t in picks.itertuples():
            rows.append([ds.image_path(ds.persons[x].images[0]) for x in (t.F, t.M, t.C)])
            ages = [ds.persons[x].age for x in (t.F, t.M, t.C)]
            age_txt = "/".join("?" if a is None else f"{a:.0f}" for a in ages)
            captions.append(f"{t.kind} {t.source}\n{t.C}\nages {age_txt}")
        out = make_contact_sheet(rows, output_dir / "triplet_check.jpg", captions,
                                 tile=int(cfg["viz"]["tile"]), column_titles=["father", "mother", "child"])
        log.info(f"Saved {out}")
    log.info(f"Done. Files in {split_dir}")


if __name__ == "__main__":
    main()
