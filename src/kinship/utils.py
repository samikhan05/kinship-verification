"""Small shared helpers: config loading, seeding, logging, JSON, images, contact sheets.

Everything here is deliberately dependency-light (numpy, PyYAML, Pillow) so that
it works on a CPU-only laptop as well as on Kaggle.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


# --------------------------------------------------------------------------- config
def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set cfg['a']['b'] = value for dotted_key 'a.b' (creating dicts as needed)."""
    node = cfg
    *parents, last = dotted_key.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[last] = value


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> dict:
    """Load the YAML config and apply 'key.sub=value' overrides.

    Override values are parsed with YAML, so '--set seed=7' gives an int and
    '--set data.use_official_triplets=false' gives a bool.
    """
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments shared by every script: config file, overrides and path shortcuts."""
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set seed=7 (repeatable)")
    parser.add_argument("--data_root", help="shortcut for --set paths.data_root=...")
    parser.add_argument("--work_dir", help="shortcut for --set paths.work_dir=...")
    parser.add_argument("--output_dir", help="shortcut for --set paths.output_dir=...")
    return parser


def config_from_args(args: argparse.Namespace) -> dict:
    """Build the final config from parsed common arguments."""
    overrides = list(args.set)
    for name in ("data_root", "work_dir", "output_dir"):
        value = getattr(args, name, None)
        if value:
            overrides.append(f"paths.{name}={value}")
    return load_config(args.config, overrides)


def get_paths(cfg: dict) -> tuple[Path, Path, Path]:
    """Return (data_root, work_dir, output_dir), creating the two writable ones."""
    p = cfg["paths"]
    data_root = Path(p["data_root"])
    work_dir = ensure_dir(p["work_dir"])
    output_dir = ensure_dir(p["output_dir"])
    return data_root, work_dir, output_dir


# --------------------------------------------------------------------------- misc
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # imported lazily: not every stage needs torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "kinship", log_file: str | Path | None = None) -> logging.Logger:
    """Logger that prints to stdout and optionally also to a file."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    if log_file is not None:
        log_file = str(log_file)
        if not any(getattr(h, "baseFilename", None) == os.path.abspath(log_file) for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logger.handlers[0].formatter)
            logger.addHandler(fh)
    return logger


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    """Make numpy scalars/arrays and Paths JSON-serialisable."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


# --------------------------------------------------------------------------- images
def load_image_rgb(path: str | Path) -> np.ndarray:
    """Read an image as an RGB uint8 array (Pillow handles odd paths better than cv2)."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def center_square(img: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to a square and resize to size x size (used for display and as a
    no-face fallback)."""
    from PIL import Image

    h, w = img.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    crop = img[top:top + s, left:left + s]
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR))


def make_contact_sheet(rows: Sequence[Sequence[str | Path | np.ndarray]],
                       out_path: str | Path,
                       captions: Sequence[str] | None = None,
                       tile: int = 112,
                       column_titles: Sequence[str] | None = None) -> Path:
    """Draw a grid of face tiles: one row per item (e.g. F | M | C), caption on the right.

    Each cell may be an image path or an RGB array. Used for triplet_check.jpg and
    for the worst false positives / negatives in the evaluation.
    """
    from PIL import Image, ImageDraw

    n_cols = max(len(r) for r in rows)
    caption_w = 260 if captions else 0
    title_h = 18 if column_titles else 0
    sheet = Image.new("RGB", (n_cols * tile + caption_w, title_h + len(rows) * tile), "white")
    draw = ImageDraw.Draw(sheet)
    if column_titles:
        for c, title in enumerate(column_titles):
            draw.text((c * tile + 4, 3), str(title), fill="black")
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            img = load_image_rgb(cell) if isinstance(cell, (str, Path)) else cell
            sheet.paste(Image.fromarray(center_square(img, tile)), (c * tile, title_h + r * tile))
        if captions:
            draw.text((n_cols * tile + 6, title_h + r * tile + 6), str(captions[r]), fill="black")
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    sheet.save(out_path, quality=90)
    return out_path


# --------------------------------------------------------------------------- caches
def restore_cached_work(work_dir: str | Path, source: str | Path | None = "auto",
                        base: str | Path = "/kaggle/input") -> Path | None:
    """Reuse an earlier run's cached embeddings (Kaggle: attach that notebook's output).

    source="auto" looks for <base>/.../work/embeddings/*/embeddings.npy a few
    levels deep. Only the small caches are copied (embeddings + attributes.csv,
    ~100 MB), not the 26k aligned crops, which are slow to copy and not needed
    for evaluation. Files that already exist in work_dir are left alone.
    Returns the source work folder, or None if nothing was found / requested.
    """
    import glob
    import shutil

    if source is None:
        return None
    if str(source) == "auto":
        hits = []
        for depth in range(1, 5):
            hits += glob.glob(str(Path(base, *["*"] * depth, "embeddings", "*", "embeddings.npy")))
        if not hits:
            return None
        # Several earlier runs may be attached: prefer the one with the most embedding
        # types (e.g. arcface AND facenet), then the most recent.
        works = {Path(h).parents[2] for h in hits}
        source = max(sorted(works), key=lambda w: (sum(Path(h).parents[2] == w for h in hits),
                                                     (w / "embeddings").stat().st_mtime))
    source, work_dir = Path(source), ensure_dir(work_dir)
    for item in ["embeddings", "attributes.csv"]:
        src, dst = source / item, work_dir / item
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst)
        elif src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
    return source
