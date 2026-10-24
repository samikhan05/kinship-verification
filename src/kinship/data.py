"""Dataset format detection, loading, and family-disjoint splits.

Two on-disk formats are supported and detected automatically:

* "fiw"    (Option A) - full Families in the Wild download:
               <root>/.../F0001/mid.csv           relationship matrix + Name + Gender
               <root>/.../F0001/MID1/*.jpg        one folder per family member
* "kaggle" (Option B) - Kaggle "Recognizing Faces in the Wild":
               <root>/train/F0001/MID1/*.jpg
               <root>/train_relationships.csv     columns p1,p2 (related pairs only,
                                                  no relationship type, no gender)

Both are loaded into the same in-memory structure, `KinshipDataset`, so the rest
of the pipeline does not care which one is on disk.

Person ids are always "F0001/MID1" (family id / member id).
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# FIW relationship codes. Cell (i, j) of mid.csv = relation of member i TO member j,
# e.g. rel[i][j] == 4 means "i is a parent of j" (and then rel[j][i] should be 1).
FIW_CODES = {1: "child", 2: "sibling", 3: "grandchild", 4: "parent", 5: "spouse",
             6: "grandparent", 7: "great-grandchild", 8: "great-grandparent", 9: "tbd"}
FIW_BLOOD_CODES = {1, 2, 3, 4, 6, 7, 8}   # spouse (5) and unknown (9) are not blood kin
FIW_INVERSE = {1: 4, 4: 1, 2: 2, 3: 6, 6: 3, 5: 5, 7: 8, 8: 7}

PID_RE = re.compile(r"(F\d{4})[/\\](MID\d+)", re.IGNORECASE)


@dataclass
class Person:
    pid: str                    # "F0001/MID1"
    fid: str                    # "F0001"
    images: list[str]           # paths relative to the dataset root (forward slashes)
    gender: str | None = None   # "m" / "f" / None (unknown)
    age: float | None = None    # estimated age (Option B, from InsightFace), else None


def pair_key(a: str, b: str) -> tuple[str, str]:
    """Order-independent key for an unordered pair of persons."""
    return (a, b) if a <= b else (b, a)


@dataclass
class KinshipDataset:
    format: str
    root: Path
    persons: dict[str, Person]
    related_pairs: set[tuple[str, str]]                    # unordered blood-relative pairs
    parent_child: set[tuple[str, str]] = field(default_factory=set)   # (parent, child), Option A
    spouses: set[tuple[str, str]] = field(default_factory=set)        # unordered, Option A
    official_triplets: list[tuple[str, str, str]] | None = None       # (F, M, C), Option A
    stats: dict = field(default_factory=dict)              # load-time counters for the report
    _adj: dict[str, set[str]] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ queries
    @property
    def families(self) -> dict[str, list[str]]:
        fams: dict[str, list[str]] = defaultdict(list)
        for pid, p in self.persons.items():
            fams[p.fid].append(pid)
        return dict(sorted(fams.items()))

    def relatives(self, pid: str) -> set[str]:
        """All blood relatives of `pid` (adjacency built once, lazily)."""
        if self._adj is None:
            adj: dict[str, set[str]] = defaultdict(set)
            for a, b in self.related_pairs:
                adj[a].add(b)
                adj[b].add(a)
            self._adj = dict(adj)
        return self._adj.get(pid, set())

    def is_related(self, a: str, b: str) -> bool:
        return pair_key(a, b) in self.related_pairs

    def image_path(self, rel: str) -> Path:
        return self.root / rel

    @property
    def n_images(self) -> int:
        return sum(len(p.images) for p in self.persons.values())


# ---------------------------------------------------------------------- detection
@lru_cache(maxsize=None)
def scan_tree(root: str, max_depth: int = 6) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Find FIW family folders (folders containing mid.csv) and loose CSV files.

    Walks the tree but never descends INTO a family folder, so the ~26k image files
    are not listed. Kaggle's /kaggle/input disk is slow at listing many files, and a
    plain recursive glob over the whole dataset took minutes there.
    Returns (family_dirs, csv_files), both sorted. Skips macOS junk.
    """
    families, csvs = [], []
    stack = [(Path(root), 0)]
    while stack:
        folder, depth = stack.pop()
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        if any(e.name == "mid.csv" for e in entries):
            families.append(folder)
            continue
        for e in entries:
            if e.name == "__MACOSX" or e.name.startswith("._"):
                continue
            if e.is_dir() and depth < max_depth:
                stack.append((Path(e.path), depth + 1))
            elif e.name.lower().endswith(".csv"):
                csvs.append(Path(e.path))
    return tuple(sorted(families)), tuple(sorted(csvs))


def find_data_root(base: str | Path = "/kaggle/input") -> Path:
    """Locate an attached dataset automatically (used when data_root is 'auto').

    Prefers full FIW (returns the folder that holds the F#### family folders),
    otherwise the Kaggle competition layout (train/ + train_relationships.csv).
    """
    base = Path(base)
    families, csvs = scan_tree(str(base), max_depth=6)
    if families:
        return families[0].parent
    for csv in csvs:
        if csv.name == "train_relationships.csv" and (csv.parent / "train").is_dir():
            return csv.parent
    raise FileNotFoundError(f"No FIW or RFIW dataset found under {base}. Attach it with Add Input.")


def resolve_data_root(value: str | Path) -> Path:
    return find_data_root() if str(value) == "auto" else Path(value)


def detect_format(root: str | Path) -> str:
    """Return 'fiw' or 'kaggle'. Option A (FIW) is preferred if both are present."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")
    if (root / "train_relationships.csv").exists() and (root / "train").is_dir():
        return "kaggle"     # checked first: it is cheap and avoids walking 20k image folders
    if scan_tree(str(root))[0]:
        return "fiw"
    raise FileNotFoundError(
        f"No known dataset format under {root}. Expected either FIW family folders "
        f"with mid.csv files, or train/ + train_relationships.csv (Kaggle RFIW).")


def load_dataset(cfg: dict) -> KinshipDataset:
    """Load the dataset named by cfg['paths']['data_root'] (format auto-detected)."""
    root = resolve_data_root(cfg["paths"]["data_root"])
    fmt = cfg["data"].get("format", "auto")
    if fmt == "auto":
        fmt = detect_format(root)
    min_imgs = int(cfg["data"].get("min_images_per_person", 1))
    if fmt == "kaggle":
        return load_kaggle(root, min_imgs)
    if fmt == "fiw":
        return load_fiw(root, min_imgs, use_official=cfg["data"].get("use_official_triplets", True))
    raise ValueError(f"Unknown data.format: {fmt}")


def is_junk(path: Path) -> bool:
    """macOS zip tools add '__MACOSX/' folders and '._name' files (metadata, not
    real data). FIW's zips contain them, e.g. a fake '._fms_pairs.csv'."""
    return "__MACOSX" in path.parts or path.name.startswith("._")


def _list_images(folder: Path, root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not is_junk(p))


def _normalise_gender(value) -> str | None:
    s = str(value).strip().lower()
    if s.startswith("m"):
        return "m"
    if s.startswith("f"):
        return "f"
    return None


# ---------------------------------------------------------------------- Option B
def load_kaggle(root: str | Path, min_images: int = 1) -> KinshipDataset:
    """Load the Kaggle RFIW layout. Related pairs whose persons have no images are dropped."""
    root = Path(root)
    persons: dict[str, Person] = {}
    for fam_dir in sorted(p for p in (root / "train").iterdir() if p.is_dir()):
        for mem_dir in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
            images = _list_images(mem_dir, root)
            if len(images) >= min_images:
                pid = f"{fam_dir.name}/{mem_dir.name}"
                persons[pid] = Person(pid=pid, fid=fam_dir.name, images=images)

    rel_df = pd.read_csv(root / "train_relationships.csv")
    pairs: set[tuple[str, str]] = set()
    dropped = 0
    for p1, p2 in zip(rel_df["p1"].astype(str).str.strip(), rel_df["p2"].astype(str).str.strip()):
        if p1 in persons and p2 in persons and p1 != p2:
            pairs.add(pair_key(p1, p2))
        else:
            dropped += 1
    stats = {"csv_rows": len(rel_df), "pairs_dropped_missing_person": dropped,
             "cross_family_pairs": sum(persons[a].fid != persons[b].fid for a, b in pairs)}
    return KinshipDataset("kaggle", root, persons, pairs, stats=stats)


# ---------------------------------------------------------------------- Option A
def parse_mid_csv(path: Path) -> tuple[list[int], dict[int, dict[int, int]], dict[int, str | None]]:
    """Parse one FIW mid.csv. Returns (member ids, rel[i][j], gender[i])."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    mids = [int(m) for m in df["MID"]]
    rel: dict[int, dict[int, int]] = {}
    for _, row in df.iterrows():
        i = int(row["MID"])
        rel[i] = {}
        for j in mids:
            if str(j) in df.columns and j != i:
                v = row[str(j)]
                rel[i][j] = int(v) if pd.notna(v) else 0
    gcol = next((c for c in df.columns if c.lower() == "gender"), None)
    gender = {int(r["MID"]): (_normalise_gender(r[gcol]) if gcol else None) for _, r in df.iterrows()}
    return mids, rel, gender


def load_fiw(root: str | Path, min_images: int = 1, use_official: bool = True) -> KinshipDataset:
    """Load the full FIW layout from every <family>/mid.csv under root."""
    root = Path(root)
    persons: dict[str, Person] = {}
    pairs: set[tuple[str, str]] = set()
    parent_child: set[tuple[str, str]] = set()
    spouses: set[tuple[str, str]] = set()
    code_counts: dict[str, int] = defaultdict(int)
    inconsistent = 0

    for fam_dir in scan_tree(str(root))[0]:
        mid_csv = fam_dir / "mid.csv"
        fid = fam_dir.name
        mids, rel, gender = parse_mid_csv(mid_csv)
        for m in mids:
            mem_dir = fam_dir / f"MID{m}"
            images = _list_images(mem_dir, root) if mem_dir.is_dir() else []
            if len(images) >= min_images:
                pid = f"{fid}/MID{m}"
                persons[pid] = Person(pid=pid, fid=fid, images=images, gender=gender.get(m))
        # Visit each unordered pair (i, j) once and read the relation "i -> j".
        # If only the mirror cell (j, i) is filled, invert it.
        for idx, i in enumerate(mids):
            for j in mids[idx + 1:]:
                code_ij = rel.get(i, {}).get(j, 0)
                code_ji = rel.get(j, {}).get(i, 0)
                if code_ij == 0 and code_ji == 0:
                    continue
                if code_ij and code_ji and FIW_INVERSE.get(code_ij) != code_ji:
                    inconsistent += 1          # the two cells disagree; trust (i, j)
                code = code_ij or FIW_INVERSE.get(code_ji, code_ji)
                code_counts[FIW_CODES.get(code, f"code{code}")] += 1
                a, b = f"{fid}/MID{i}", f"{fid}/MID{j}"
                if a not in persons or b not in persons:
                    continue                   # member without images
                if code in FIW_BLOOD_CODES:
                    pairs.add(pair_key(a, b))
                if code == 4:                  # i is parent of j
                    parent_child.add((a, b))
                elif code == 1:                # i is child of j
                    parent_child.add((b, a))
                elif code == 5:
                    spouses.add(pair_key(a, b))

    official = find_official_triplets(root, persons) if use_official else None
    stats = {"relationship_codes": dict(code_counts), "matrix_inconsistent_cells": inconsistent,
             "official_triplets": None if official is None else len(official)}
    return KinshipDataset("fiw", root, persons, pairs, parent_child, spouses, official, stats)


def find_official_triplets(root: Path, persons: dict[str, Person]) -> list[tuple[str, str, str]] | None:
    """Best-effort reader for official tri-subject (FMS/FMD) lists shipped with FIW.

    Looks for CSV files whose name mentions fms / fmd / tri. In each row it finds
    the first three "F####/MID#" ids (in column order: father, mother, child).
    Rows with a label column equal to 0 are skipped (we build our own negatives).
    Returns None if no list was found or nothing could be parsed.
    """
    candidates = [p for p in scan_tree(str(root))[1]
                  if re.search(r"fms|fmd|tri", p.stem, re.IGNORECASE)]
    triplets: set[tuple[str, str, str]] = set()
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        label_col = next((c for c in df.columns if str(c).strip().lower() in ("label", "labels", "y")), None)
        for _, row in df.iterrows():
            if label_col is not None and str(row[label_col]).strip() in ("0", "0.0", "False"):
                continue
            ids = []
            for v in row.values:
                m = PID_RE.search(str(v))
                if m:
                    ids.append(f"{m.group(1).upper()}/{m.group(2).upper()}")
            if len(ids) >= 3 and all(i in persons for i in ids[:3]) and len(set(ids[:3])) == 3:
                f, mo, c = ids[:3]
                # Respect gender if known: father must not be female, mother not male.
                if persons[f].gender == "f" and persons[mo].gender == "m":
                    f, mo = mo, f
                triplets.add((f, mo, c))
    return sorted(triplets) if triplets else None


# ---------------------------------------------------------------------- attributes
def apply_attributes(ds: KinshipDataset, attrs: pd.DataFrame | None) -> int:
    """Copy estimated gender/age (columns pid, gender, age) onto persons.

    Ground-truth gender (FIW) is never overwritten; age is always taken from the
    estimate because FIW has no ages. Returns how many persons received attributes.
    """
    if attrs is None:
        return 0
    n = 0
    for row in attrs.itertuples(index=False):
        p = ds.persons.get(str(row.pid))
        if p is None:
            continue
        if p.gender is None and isinstance(row.gender, str):
            p.gender = _normalise_gender(row.gender)
        if pd.notna(row.age):
            p.age = float(row.age)
        n += 1
    return n


# ---------------------------------------------------------------------- splits
def split_families(fids: list[str], ratios: dict[str, float], seed: int,
                   strata: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Assign whole families to train / val / test (family-disjoint by construction).

    `strata` (optional) maps fid -> group name; each group is split separately with
    the same ratios. We use this so that families that contain triplets are spread
    70/15/15 too, rather than leaving it to chance.
    """
    names = list(ratios)
    total = sum(ratios.values())
    fracs = np.array([ratios[n] / total for n in names])
    groups: dict[str, list[str]] = defaultdict(list)
    for fid in sorted(set(fids)):
        groups[(strata or {}).get(fid, "all")].append(fid)

    rng = np.random.default_rng(seed)
    out: dict[str, list[str]] = {n: [] for n in names}
    for gname in sorted(groups):
        members = np.array(groups[gname])
        rng.shuffle(members)
        # Cumulative rounding: sizes always add up to len(members).
        bounds = np.round(np.cumsum(fracs) * len(members)).astype(int)
        start = 0
        for name, end in zip(names, bounds):
            out[name].extend(members[start:end].tolist())
            start = end
    return {n: sorted(v) for n, v in out.items()}
