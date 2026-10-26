"""Report figures (matplotlib, PNG): ROC curves, score histograms, confusion matrix,
attention-weight distributions.

Colours: a fixed categorical order validated for colour-blind separation; each
line also gets its own dash pattern, so identity never relies on colour alone.
Every figure is drawn from the score files saved by the evaluation, never from
numbers typed by hand.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # no display needed (Kaggle / scripts)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # blue, orange, aqua, yellow, magenta
DASHES = ["-", "--", "-.", ":", (0, (6, 2, 1, 2))]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUES = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def _figure(w: float = 6.4, h: float = 4.4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    _style(fig, ax)
    return fig, ax


def _style(fig, ax) -> None:
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT2)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def _save(fig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def roc_plot(curves: list[tuple[str, np.ndarray, np.ndarray]], out, title: str) -> Path:
    """One ROC curve per experiment (at most 5), with AUC in the legend."""
    from sklearn.metrics import roc_auc_score

    fig, ax = _figure(6.4, 5.2)
    ax.plot([0, 1], [0, 1], color=TEXT2, linewidth=1, linestyle=(0, (2, 3)), label="chance")
    for i, (name, y, s) in enumerate(curves[:5]):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=SERIES[i], linestyle=DASHES[i], linewidth=2,
                label=f"{name}  (AUC {roc_auc_score(y, s):.3f})")
    ax.axvline(0.1, color=TEXT2, linewidth=0.8, alpha=0.6)
    ax.text(0.105, 0.03, "FAR = 0.1", color=TEXT2, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False accept rate (non-kin accepted)")
    ax.set_ylabel("True accept rate (kin accepted)")
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=TEXT)
    return _save(fig, out)


def score_histogram(y: np.ndarray, s: np.ndarray, threshold: float, out, title: str) -> Path:
    """Scores of kin vs non-kin triplets, with the decision threshold (chosen on validation)."""
    fig, ax = _figure()
    bins = np.linspace(min(s.min(), threshold), max(s.max(), threshold), 31)
    ax.hist(s[y == 1], bins=bins, color=SERIES[0], alpha=0.75, label=f"kin (n={int((y == 1).sum())})",
            edgecolor=SURFACE, linewidth=1)
    ax.hist(s[y == 0], bins=bins, color=SERIES[1], alpha=0.65, label=f"non-kin (n={int((y == 0).sum())})",
            edgecolor=SURFACE, linewidth=1, hatch="//")
    ax.axvline(threshold, color=TEXT, linewidth=1.2, linestyle="--")
    ax.text(threshold, ax.get_ylim()[1] * 0.95, f"  threshold {threshold:.2f}", color=TEXT, fontsize=8, va="top")
    ax.set_xlabel("Model score (higher = more likely their child)")
    ax.set_ylabel("Triplets")
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    return _save(fig, out)


def confusion_matrix_plot(y: np.ndarray, s: np.ndarray, threshold: float, out, title: str) -> Path:
    """2 x 2 counts at the validation threshold."""
    pred = (s >= threshold).astype(int)
    m = np.array([[np.sum((y == 1) & (pred == 1)), np.sum((y == 1) & (pred == 0))],
                  [np.sum((y == 0) & (pred == 1)), np.sum((y == 0) & (pred == 0))]])
    fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    ax.imshow(m, cmap=cmap, vmin=0, vmax=m.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]}\n{m[i, j] / m[i].sum():.0%} of row", ha="center", va="center",
                    color="white" if m[i, j] > m.max() * 0.55 else TEXT, fontsize=10)
    ax.set_xticks([0, 1], ["predicted kin", "predicted non-kin"])
    ax.set_yticks([0, 1], ["actually\nkin", "actually\nnon-kin"])
    ax.tick_params(colors=TEXT, labelsize=9, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    return _save(fig, out)


def alpha_plot(panels: list[tuple[str, np.ndarray, np.ndarray]], out, title: str) -> Path:
    """Distribution of alpha_F (weight on the father) for sons vs daughters, one panel
    per model. panels: (model name, alpha_F array, kind array of 'FMS'/'FMD')."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), dpi=150, sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (name, alpha, kind) in zip(axes, panels):
        _style(fig, ax)
        bins = np.linspace(0, 1, 26)
        for i, (k, label) in enumerate((("FMS", "sons"), ("FMD", "daughters"))):
            a = alpha[kind == k]
            ax.hist(a, bins=bins, color=SERIES[i], alpha=0.7, edgecolor=SURFACE, linewidth=1,
                    hatch=None if i == 0 else "//", label=f"{label} (mean {a.mean():.2f})")
        ax.axvline(0.5, color=TEXT2, linewidth=1, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)          # head-room so the legend never covers bars
        ax.set_xlabel("alpha_F  (0 = all mother, 1 = all father)")
        ax.set_ylabel("Kin test triplets")
        ax.set_title(name, color=TEXT, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT, loc="upper left")
    fig.suptitle(title, color=TEXT, fontsize=11, x=0.01, ha="left")
    return _save(fig, out)
