"""Tiny synthetic family datasets for CPU smoke tests and unit tests.

The "faces" are just small coloured images; what matters is the folder layout and
the relationship files, which mimic the two real formats exactly. A child's colour
is a mix of its parents' colours, so there is a weak, known kinship signal. Ground-truth
gender and age are also written to `attributes_gt.csv`, standing in for what
InsightFace would estimate on real data.

Each synthetic family has: father, mother, 1-3 children, and (half the time) the
father's parents. That lets us test that grandparents are not mistaken for parents.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import FIW_INVERSE


def _make_family(rng: np.random.Generator) -> tuple[list[dict], list[tuple[int, int, int]]]:
    """Return members [{mid, role, gender, age}] and relations [(i, j, code)] meaning
    "i is <code> of j" using FIW codes (4 parent, 2 sibling, 6 grandparent, 5 spouse)."""
    members = [{"mid": 1, "role": "father", "gender": "m", "age": 45 + rng.normal(0, 3)},
               {"mid": 2, "role": "mother", "gender": "f", "age": 43 + rng.normal(0, 3)}]
    rels = [(1, 2, 5)]
    for _ in range(int(rng.integers(1, 4))):
        mid = len(members) + 1
        members.append({"mid": mid, "role": "child", "gender": str(rng.choice(["m", "f"])),
                        "age": float(rng.uniform(5, 20))})
        rels += [(1, mid, 4), (2, mid, 4)]
    children = [m["mid"] for m in members if m["role"] == "child"]
    for a_i, a in enumerate(children):
        for b in children[a_i + 1:]:
            rels.append((a, b, 2))
    if rng.random() < 0.5:   # paternal grandparents
        gf, gm = len(members) + 1, len(members) + 2
        members += [{"mid": gf, "role": "grandfather", "gender": "m", "age": 72 + rng.normal(0, 3)},
                    {"mid": gm, "role": "grandmother", "gender": "f", "age": 70 + rng.normal(0, 3)}]
        rels += [(gf, gm, 5), (gf, 1, 4), (gm, 1, 4)]
        rels += [(g, c, 6) for g in (gf, gm) for c in children]
    return members, rels


def _inherited_colours(members: list[dict], rels: list[tuple[int, int, int]],
                       rng: np.random.Generator) -> dict[int, np.ndarray]:
    """Give each person a base colour; a child's colour is the mix of its parents'
    plus noise. This plants a weak, known "kinship signal" in the fake images, so
    smoke tests can check that the pipeline scores kin above non-kin."""
    parents: dict[int, list[int]] = {}
    for i, j, code in rels:
        if code == 4:
            parents.setdefault(j, []).append(i)
    colours: dict[int, np.ndarray] = {}

    def colour(mid: int) -> np.ndarray:
        if mid not in colours:
            ps = parents.get(mid)
            if ps:
                base = np.mean([colour(p) for p in ps], axis=0) + rng.normal(0, 25, size=3)
            else:
                base = rng.integers(0, 256, size=3).astype(float)
            colours[mid] = np.clip(base, 0, 255)
        return colours[mid]

    for m in members:
        colour(m["mid"])
    return colours


def _true_triplets(fid: str, members: list[dict], rels: list[tuple[int, int, int]]) -> list[tuple]:
    """(F, M, C) for every member with a known father and mother. Note that the
    father himself is a true child of the grandparents."""
    gender = {m["mid"]: m["gender"] for m in members}
    parents: dict[int, list[int]] = {}
    for i, j, code in rels:
        if code == 4:
            parents.setdefault(j, []).append(i)
    out = []
    for child, ps in parents.items():
        for f in (p for p in ps if gender[p] == "m"):
            for m in (p for p in ps if gender[p] == "f"):
                out.append((f"{fid}/MID{f}", f"{fid}/MID{m}", f"{fid}/MID{child}"))
    return out


def make_synthetic(root: str | Path, fmt: str = "kaggle", n_families: int = 40,
                   images_per_person: tuple[int, int] = (2, 4), size: int = 64,
                   seed: int = 0) -> dict:
    """Write a synthetic dataset in 'kaggle' or 'fiw' layout under `root`.

    Returns ground truth: {"triplets": [(F, M, C)], "persons": {pid: {...}}}.
    """
    from PIL import Image

    root = Path(root)
    rng = np.random.default_rng(seed)
    base = root / ("train" if fmt == "kaggle" else "FIDs")
    pair_rows, attr_rows, truth_triplets, persons = [], [], [], {}

    next_photo = 0
    for f in range(1, n_families + 1):
        fid = f"F{f:04d}"
        members, rels = _make_family(rng)
        colours = _inherited_colours(members, rels, rng)
        # Like FIW: file names are <photo>_face<k>.jpg. Each family has a few group
        # photos that several members appear in, plus solo photos.
        group_photos = list(range(next_photo, next_photo + 3))
        next_photo += 3
        for m in members:
            pid = f"{fid}/MID{m['mid']}"
            persons[pid] = {**m, "fid": fid}
            attr_rows.append({"pid": pid, "gender": m["gender"], "age": round(m["age"], 1)})
            folder = base / fid / f"MID{m['mid']}"
            folder.mkdir(parents=True, exist_ok=True)
            colour = colours[m["mid"]]
            unused_groups = list(group_photos)
            for _ in range(int(rng.integers(images_per_person[0], images_per_person[1] + 1))):
                if unused_groups and rng.random() < 0.5:
                    photo = unused_groups.pop(int(rng.integers(len(unused_groups))))
                else:
                    photo, next_photo = next_photo, next_photo + 1
                noise = rng.integers(-30, 30, size=(size, size, 3))
                img = np.clip(colour + noise, 0, 255).astype(np.uint8)
                # face index = member id, so two members in one photo get different names
                Image.fromarray(img).save(folder / f"P{photo:05d}_face{m['mid']}.jpg")
        truth_triplets += _true_triplets(fid, members, rels)

        if fmt == "kaggle":
            # Only blood relatives are listed (no spouses), as in train_relationships.csv.
            pair_rows += [{"p1": f"{fid}/MID{i}", "p2": f"{fid}/MID{j}"}
                          for i, j, code in rels if code != 5]
        else:
            n = len(members)
            mat = np.zeros((n, n), dtype=int)
            for i, j, code in rels:
                mat[i - 1, j - 1] = code
                mat[j - 1, i - 1] = FIW_INVERSE[code]
            df = pd.DataFrame(mat, columns=[str(i) for i in range(1, n + 1)])
            df.insert(0, "MID", range(1, n + 1))
            df["Name"] = [m["role"] for m in members]
            df["Gender"] = ["male" if m["gender"] == "m" else "female" for m in members]
            df.to_csv(base / fid / "mid.csv", index=False)

    if fmt == "kaggle":
        pd.DataFrame(pair_rows).to_csv(root / "train_relationships.csv", index=False)
    pd.DataFrame(attr_rows).to_csv(root / "attributes_gt.csv", index=False)
    return {"triplets": truth_triplets, "persons": persons}
