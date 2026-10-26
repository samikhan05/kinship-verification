"""Build outputs/results.md and the report figures from the saved results - never typed by hand.

    python scripts/make_report.py
    # locally, from a downloaded Kaggle outputs.zip, also copying the report into docs/results/:
    python scripts/make_report.py --output_dir kaggle_outputs/outputs --work_dir work/fiw \
        --data_root data/fiw --copy_to docs/results

Figures (outputs/figures/): ROC curves of the main experiments, score histogram and
confusion matrix of the best model, attention-weight distributions (sons vs
daughters), and contact sheets of the 10 worst false positives / false negatives
(needs the dataset images; skipped if they are not available).
"""
import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kinship.utils import (add_common_args, config_from_args, get_paths, load_json,  # noqa: E402
                           make_contact_sheet)

TAR = "tar@far=0.1"
HEAD = "template_xphoto"   # headline protocol (see src/kinship/evaluate.py)


def fmt(x) -> str:
    return "–" if x is None or x != x else f"{x:.3f}"   # x != x is True for NaN


def exp_sort_key(name: str):
    """E0-mean, E0-max, E1-LR, ..., E2, E3: by experiment number, then name."""
    m = re.match(r"[A-Z](\d+)", name)
    return (int(m.group(1)) if m else 99, name)


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(r) + " |" for r in rows]


ROC_EXPERIMENTS = ["E0-mean", "E1-LR", "E2-hardneg", "E2b", "E3"]   # at most 5 lines per chart
BEST = "E2-hardneg"                                                   # best headline AUC on the test set


def load_scores(output_dir: Path, name: str, set_name: str = "test", proto: str = HEAD) -> pd.DataFrame | None:
    path = output_dir / "scores" / f"{name}__{set_name}__{proto}.csv"
    return pd.read_csv(path) if path.exists() else None


def attention_section(trip: list[dict]) -> list[str]:
    """Table of the attention analysis (does alpha follow the parent the child looks more like?)."""
    rows = []
    for r in trip:
        a = r.get("alpha_analysis")
        if a:
            rows.append([r["experiment"], f"{a['spearman_alphaF_vs_cos_gap']:+.2f}", f"{a['spearman_p_value']:.1e}",
                         f"{a['share_where_alpha_picks_more_similar_parent']:.0%}", f"{a['mean_alpha_F']:.2f}",
                         f"{a.get('mean_alpha_F_FMS', float('nan')):.2f}", f"{a.get('mean_alpha_F_FMD', float('nan')):.2f}"])
    if not rows:
        return []
    return ["## Attention weights (kin test triplets)", "",
            "Does alpha_F rise when the child is more similar to the father than to the mother "
            "(cos(F,C) - cos(M,C))? Only where it does can alpha be read as \"resembles father X%\".", ""] + \
        table(["Exp", "Spearman rho", "p-value", "alpha picks more similar parent", "mean alpha_F",
               "mean alpha_F (sons)", "mean alpha_F (daughters)"], rows) + [""]


def figures_section(cfg: dict, output_dir: Path, trip: list[dict]) -> list[str]:
    from kinship import plots

    fig_dir = output_dir / "figures"
    names = {r["experiment"]: r for r in trip}
    lines = ["## Figures", ""]

    curves = []
    for name in ROC_EXPERIMENTS:
        df = load_scores(output_dir, name)
        if df is not None:
            curves.append((name, df["label"].to_numpy(), df["score"].to_numpy()))
    if curves:
        plots.roc_plot(curves, fig_dir / "roc_test.png", "ROC curves, test set (templates, different photos)")
        lines += ["![ROC curves](figures/roc_test.png)", ""]

    best = load_scores(output_dir, BEST)
    if best is not None and BEST in names:
        thr = names[BEST]["modes"][HEAD]["test"]["threshold"]
        y, sc = best["label"].to_numpy(), best["score"].to_numpy()
        plots.score_histogram(y, sc, thr, fig_dir / "score_hist_best.png", f"{BEST}: test scores, kin vs non-kin")
        plots.confusion_matrix_plot(y, sc, thr, fig_dir / "confusion_best.png", f"{BEST}: test set at the val threshold")
        lines += [f"![Score histogram](figures/score_hist_best.png) ![Confusion matrix](figures/confusion_best.png)", ""]

    panels = []
    for name in (BEST, "E3"):
        df = load_scores(output_dir, name)
        if df is not None and "alpha_F" in df:
            kin = df[df["label"] == 1]
            panels.append((name, kin["alpha_F"].to_numpy(), kin["kind"].to_numpy()))
    if panels:
        plots.alpha_plot(panels, fig_dir / "alpha_distribution.png", "Attention weight on the father, kin test triplets")
        lines += ["![Attention weights](figures/alpha_distribution.png)", ""]

    if best is not None:
        lines += contact_sheets(cfg, best, names[BEST]["modes"][HEAD]["test"]["threshold"], fig_dir)
    return lines


def contact_sheets(cfg: dict, df: pd.DataFrame, thr: float, fig_dir: Path) -> list[str]:
    """10 worst false positives (non-kin with the highest scores) and false negatives
    (kin with the lowest scores) of the best model, shown as F | M | C."""
    try:
        from kinship.data import load_dataset

        ds = load_dataset(cfg)
    except Exception as e:                       # images not available here
        return [f"*Error contact sheets skipped: dataset images not available ({e.__class__.__name__}).*", ""]
    out = []
    for kind, rows, label in (("fp", df[(df["label"] == 0) & (df["score"] >= thr)].nlargest(10, "score"),
                               "worst false positives (non-kin scored as kin)"),
                              ("fn", df[(df["label"] == 1) & (df["score"] < thr)].nsmallest(10, "score"),
                               "worst false negatives (kin scored as non-kin)")):
        if rows.empty:
            continue
        images = [[ds.image_path(ds.persons[p].images[0]) for p in (t.F, t.M, t.C)] for t in rows.itertuples()]
        caps = [f"score {t.score:.2f}  {t.kind}  {t.neg_type}\n{t.C}" for t in rows.itertuples()]
        make_contact_sheet(images, fig_dir / f"worst_{kind}.jpg", caps, tile=96,
                           column_titles=["father", "mother", "child"])
        out += [f"**{BEST}, {label}:**", "", f"![{label}](figures/worst_{kind}.jpg)", ""]
    return out


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--copy_to", help="also copy results.md and figures/ to this folder (e.g. docs/results)")
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
        n_test = trip[0]["modes"].get(HEAD, {}).get("test", {})
        lines += ["## Tri-subject verification: main test set", "",
                  "**Headline protocol: per-person templates, parent and child compared only through images "
                  "from different photos.** In FIW, relatives are often cut from the same photo, and same-photo "
                  "faces look alike for non-kinship reasons (lighting, camera, colour). The *all photos* "
                  "columns show the score when that shortcut is allowed.", "",
                  f"Threshold chosen on validation only. Test set: {n_test.get('n', '?')} triplets, 50% kin / 50% "
                  f"non-kin; {n_test.get('dropped_rows', 0)} rows were dropped because a parent and the child "
                  "never appear in different photos.", ""]
        rows = []
        for r in trip:
            m, a = r["modes"].get(HEAD, r["modes"]["template"])["test"], r["modes"]["template"]["test"]
            rows.append([r["experiment"], r["description"], fmt(m["accuracy"]), fmt(m["roc_auc"]),
                         fmt(m["f1"]), fmt(m.get(TAR)), fmt(a["accuracy"]), fmt(a["roc_auc"])])
        lines += table(["Exp", "Description", "Accuracy", "ROC-AUC", "F1", "TAR@FAR=0.1",
                        "Acc (all photos)", "AUC (all photos)"], rows) + [""]

        lines += ["## Breakdowns (test, different-photo protocol)", "",
                  "*Per-image*: one random photo per person instead of the average over all photos. "
                  "*Hard*: negatives with one wrong parent (the child is real). ", ""]
        rows = []
        for r in trip:
            tm = r["modes"].get(HEAD, r["modes"]["template"])
            im = r["modes"].get("image_xphoto", r["modes"]["image"])
            hard = tm.get("test_hard", {})
            rows.append([r["experiment"], fmt(im["test"]["accuracy"]), fmt(im["test"]["roc_auc"]),
                         fmt(hard.get("accuracy")), fmt(hard.get("roc_auc")),
                         fmt(tm.get("test_FMS", {}).get("accuracy")), fmt(tm.get("test_FMD", {}).get("accuracy")),
                         fmt(tm["val"]["accuracy"]), fmt(tm["val"]["roc_auc"])])
        lines += table(["Exp", "Acc (per-image)", "AUC (per-image)", "Acc (hard)", "AUC (hard)",
                        "Acc FMS", "Acc FMD", "Val acc", "Val AUC"], rows) + [""]

    for r in pair:
        tm = r["modes"].get(HEAD, r["modes"]["template"])
        am = r["modes"]["template"]
        lines += [f"## Pairwise parent-child verification: {r['experiment']}", "",
                  f"{r['description']}. Templates; different-photo protocol unless marked *all photos*.", ""]
        rels = ["test"] + sorted(k for k in tm if k.startswith("test_"))
        rows = [[k.replace("test_", "") if k != "test" else "all", str(tm[k]["n"]), fmt(tm[k]["accuracy"]),
                 fmt(tm[k]["roc_auc"]), fmt(am.get(k, {}).get("roc_auc"))] for k in rels]
        lines += table(["Relation", "n", "Accuracy", "ROC-AUC", "AUC (all photos)"], rows) + [""]

    lines += attention_section(trip)
    lines += figures_section(cfg, output_dir, trip)

    lines += ["## Reference", "",
              "Published RFIW tri-subject leaderboard accuracies are roughly 0.68-0.79. Our split and "
              "protocol differ, so they are a reference point, not a direct comparison.", ""]
    out = output_dir / "results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out}")
    if args.copy_to:
        dest = Path(args.copy_to)
        (dest / "figures").mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, dest / "results.md")
        for fig in (output_dir / "figures").glob("*"):
            shutil.copy2(fig, dest / "figures" / fig.name)
        print(f"Copied results.md and figures to {dest}")


if __name__ == "__main__":
    main()
