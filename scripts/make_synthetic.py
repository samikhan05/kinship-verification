"""Write a small synthetic dataset for local CPU smoke tests.

    python scripts/make_synthetic.py --out data/synth_kaggle --format kaggle
    python scripts/make_synthetic.py --out data/synth_fiw --format fiw
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kinship.synthetic import make_synthetic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["kaggle", "fiw"], default="kaggle")
    ap.add_argument("--families", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    truth = make_synthetic(args.out, args.format, n_families=args.families, seed=args.seed)
    print(f"Wrote {args.format} dataset to {args.out}: {len(truth['persons'])} persons, "
          f"{len(truth['triplets'])} true triplets. Ground-truth attributes: "
          f"{Path(args.out) / 'attributes_gt.csv'}")


if __name__ == "__main__":
    main()
