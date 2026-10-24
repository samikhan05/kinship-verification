"""Step 0: look at the dataset before doing anything else.

Prints and saves (outputs/data_summary.md): detected format, number of families,
persons, images, images per person, related pairs, image sizes, and a guess of
whether the images are already face crops.

    python scripts/inspect_data.py                       # Kaggle paths from the config
    python scripts/inspect_data.py --data_root data/synth_kaggle --output_dir outputs
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kinship.data import find_files, load_dataset  # noqa: E402
from kinship.utils import add_common_args, config_from_args, get_paths  # noqa: E402


def image_size_stats(ds, n_sample: int, seed: int) -> dict:
    """Open a random sample of images (header only) and summarise their sizes."""
    from PIL import Image

    all_imgs = [img for p in ds.persons.values() for img in p.images]
    rng = np.random.default_rng(seed)
    sample = rng.choice(all_imgs, size=min(n_sample, len(all_imgs)), replace=False)
    sizes = []
    for rel in sample:
        with Image.open(ds.image_path(rel)) as im:
            sizes.append(im.size)  # (width, height)
    w = np.array([s[0] for s in sizes])
    h = np.array([s[1] for s in sizes])
    aspect = w / h
    common = Counter(sizes).most_common(3)
    # Heuristic only: face crops in RFIW/FIW are small and roughly square.
    # A detector-based check is added in M2 (extract_embeddings logs detection rates).
    looks_cropped = bool(np.median(np.maximum(w, h)) <= 256 and 0.8 <= np.median(aspect) <= 1.25)
    return {"sampled": len(sizes), "width_min_med_max": (int(w.min()), int(np.median(w)), int(w.max())),
            "height_min_med_max": (int(h.min()), int(np.median(h)), int(h.max())),
            "most_common_sizes": [f"{a}x{b} ({n})" for (a, b), n in common],
            "likely_face_crops": looks_cropped}


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    args = ap.parse_args()
    cfg = config_from_args(args)
    _, _, output_dir = get_paths(cfg)

    ds = load_dataset(cfg)
    per_person = np.array([len(p.images) for p in ds.persons.values()])
    lines = ["# Data summary", "",
             f"- Data root: `{ds.root}`",
             f"- Format detected: **{ds.format}** "
             f"({'Option A: full FIW' if ds.format == 'fiw' else 'Option B: Kaggle RFIW'})",
             f"- Families: {len(ds.families)}",
             f"- Persons (with >= {cfg['data']['min_images_per_person']} image): {len(ds.persons)}",
             f"- Images: {ds.n_images}",
             f"- Images per person (min / median / max): {per_person.min()} / "
             f"{np.median(per_person):g} / {per_person.max()}",
             f"- Related (blood-relative) pairs: {len(ds.related_pairs)}"]

    with_rel = {x for pair in ds.related_pairs for x in pair}
    lines.append(f"- Persons that appear in at least one related pair: {len(with_rel)}")
    if ds.format == "fiw":
        genders = Counter(p.gender or "unknown" for p in ds.persons.values())
        lines += [f"- Parent-child pairs: {len(ds.parent_child)}",
                  f"- Spouse pairs: {len(ds.spouses)}",
                  f"- Gender counts: {dict(genders)}",
                  f"- Relationship codes found: {ds.stats['relationship_codes']}",
                  f"- Matrix cells that disagree with their mirror cell: {ds.stats['matrix_inconsistent_cells']}",
                  f"- Official tri-subject (FMS/FMD) triplets found: {ds.stats['official_triplets']}"]
    else:
        lines += [f"- Rows in train_relationships.csv: {ds.stats['csv_rows']}",
                  f"- Pairs dropped (person folder missing or empty): {ds.stats['pairs_dropped_missing_person']}",
                  f"- Pairs that cross two families: {ds.stats['cross_family_pairs']}",
                  "- Gender / age: not provided, estimated with InsightFace in extract_embeddings.py"]

    sizes = image_size_stats(ds, int(cfg["data"]["inspect_sample_images"]), int(cfg["seed"]))
    lines += ["", "## Image sizes (random sample)", "",
              f"- Images opened: {sizes['sampled']}",
              f"- Width min / median / max: {sizes['width_min_med_max']}",
              f"- Height min / median / max: {sizes['height_min_med_max']}",
              f"- Most common sizes: {', '.join(sizes['most_common_sizes'])}",
              f"- Already face crops? {'likely yes' if sizes['likely_face_crops'] else 'probably not'} "
              "(heuristic: small and square; detection rate is checked during alignment)"]

    if ds.format == "fiw":
        # Show one family's matrix so the "row i is <code> of column j" convention
        # can be checked by eye against the member names.
        first_csv = find_files(ds.root, "mid.csv")[0]
        lines += ["", f"## Example relationship matrix ({first_csv.parent.name})", "",
                  "Cell (row i, column j) is read as *member i is <code> of member j*: "
                  "1 child, 2 sibling, 3 grandchild, 4 parent, 5 spouse, 6 grandparent. "
                  "Check this against the names below.", "", "```",
                  pd.read_csv(first_csv).to_string(index=False), "```"]

    text = "\n".join(lines) + "\n"
    print(text)
    out = output_dir / "data_summary.md"
    out.write_text(text, encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
