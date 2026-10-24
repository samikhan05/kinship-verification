"""Build outputs/results.md from outputs/results/*.json - never typed by hand.

    python scripts/make_report.py
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kinship.utils import add_common_args, config_from_args, get_paths, load_json  # noqa: E402

TAR = "tar@far=0.1"


def fmt(x) -> str:
    return "–" if x is None or x != x else f"{x:.3f}"   # x != x is True for NaN


def exp_sort_key(name: str):
    """E0-mean, E0-max, E1-LR, ..., E2, E3: by experiment number, then name."""
    m = re.match(r"[A-Z](\d+)", name)
    return (int(m.group(1)) if m else 99, name)


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(r) + " |" for r in rows]


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = config_from_args(args)
    _, work_dir, output_dir = get_paths(cfg)

    results = [load_json(p) for p in sorted((output_dir / "results").glob("*.json"))]
    trip = sorted([r for r in results if r.get("task") != "pairwise"], key=lambda r: exp_sort_key(r["experiment"]))
    pair = [r for r in results if r.get("task") == "pairwise"]

    lines = ["# Results", "",
             f"*Generated automatically by `scripts/make_report.py` on {datetime.now():%Y-%m-%d %H:%M} "
             "from `outputs/results/*.json`. Do not edit by hand.*", ""]

    summary_path = work_dir / "splits" / "summary.json"
    if summary_path.exists():
        s = load_json(summary_path)
        lines += [f"Dataset format: **{s['format']}**, positive triplets from **{s['positives']['source']}**."]
        for split, c in s["splits"].items():
            lines.append(f"- {split}: {c['families']} families, {c['positive_triplets']} positive triplets "
                         f"({c['kinds']}), balanced set {c['triplets_after_balance']}, "
                         f"hard set {c['hard_triplets']}, pairs {c['pairs']}")
        lines.append("")
    det_path = output_dir / "detection_arcface.json"
    if det_path.exists():
        d = load_json(det_path)
        lines += [f"Face detection: {d['detected']}/{d['images']} images ({d['detection_rate']:.1%}); "
                  f"{d['fallback_centre_crop']} centre-crop fallbacks; {d['multiple_faces']} with several faces.", ""]

    if not trip:
        lines += ["No experiments have been run yet."]
    else:
        lines += ["## Tri-subject verification: main test set (per-person template scoring)", "",
                  "Threshold chosen on validation only. Test sets are 50% kin / 50% non-kin.", ""]
        rows = []
        for r in trip:
            m = r["modes"]["template"]["test"]
            rows.append([r["experiment"], r["description"], fmt(m["accuracy"]), fmt(m["roc_auc"]),
                         fmt(m["f1"]), fmt(m.get(TAR))])
        lines += table(["Exp", "Description", "Accuracy", "ROC-AUC", "F1", "TAR@FAR=0.1"], rows) + [""]

        lines += ["## Breakdowns (test)", "",
                  "*Per-image*: one random photo per person instead of the average over all photos. "
                  "*Hard*: negatives with one wrong parent (the child is real). ", ""]
        rows = []
        for r in trip:
            tm, im = r["modes"]["template"], r["modes"]["image"]
            hard = tm.get("test_hard", {})
            rows.append([r["experiment"], fmt(im["test"]["accuracy"]), fmt(im["test"]["roc_auc"]),
                         fmt(hard.get("accuracy")), fmt(hard.get("roc_auc")),
                         fmt(tm.get("test_FMS", {}).get("accuracy")), fmt(tm.get("test_FMD", {}).get("accuracy")),
                         fmt(tm["val"]["accuracy"]), fmt(tm["val"]["roc_auc"])])
        lines += table(["Exp", "Acc (per-image)", "AUC (per-image)", "Acc (hard)", "AUC (hard)",
                        "Acc FMS", "Acc FMD", "Val acc", "Val AUC"], rows) + [""]

    for r in pair:
        tm = r["modes"]["template"]
        lines += [f"## Pairwise parent-child verification: {r['experiment']}", "", r["description"], ""]
        rels = ["test"] + sorted(k for k in tm if k.startswith("test_"))
        rows = [[k.replace("test_", "") if k != "test" else "all", str(tm[k]["n"]), fmt(tm[k]["accuracy"]),
                 fmt(tm[k]["roc_auc"])] for k in rels]
        lines += table(["Relation", "n", "Accuracy", "ROC-AUC"], rows) + [""]

    lines += ["## Reference", "",
              "Published RFIW tri-subject leaderboard accuracies are roughly 0.68-0.79. Our split and "
              "protocol differ, so they are a reference point, not a direct comparison.", ""]
    out = output_dir / "results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
