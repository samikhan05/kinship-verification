"""Building (father, mother, child) triplets, negatives and pairwise pairs.

Positives
    Option A (FIW): F = male parent of C, M = female parent of C, taken from the
        relationship matrix (or from official FMS/FMD lists when present).
    Option B (Kaggle): "pseudo-triplets" mined from related pairs + estimated
        gender/age (see `mine_pseudo_triplets`). These labels are noisy.

Negatives (always drawn from the SAME split, from a DIFFERENT family)
    main  : (F, M, C')  C' = stranger with the same gender as C and a similar age.
    hard  : (F, M', C) or (F', M, C) - one parent swapped for an unrelated adult of
            the same gender. Reported separately, it tests "these are BOTH his parents".
    Every positive gets exactly one negative, so each set is 50% / 50%.

All randomness comes from numpy Generators seeded from cfg['seed'], so the same
config always produces the same files.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .data import KinshipDataset, pair_key, split_families

TRIPLET_COLUMNS = ["F", "M", "C", "label", "kind", "neg_type", "fid", "source"]
PAIR_COLUMNS = ["p1", "p2", "label", "rel", "fid"]
SPLITS = ("train", "val", "test")


def kind_of(child_gender: str | None) -> str:
    """FMS = father-mother-son, FMD = father-mother-daughter."""
    return {"m": "FMS", "f": "FMD"}.get(child_gender, "FM?")


# ---------------------------------------------------------------------- positives
def triplets_from_parent_child(ds: KinshipDataset) -> tuple[list[dict], dict]:
    """Option A: every child with a known male AND a known female parent."""
    parents_of: dict[str, set[str]] = defaultdict(set)
    for parent, child in ds.parent_child:
        parents_of[child].add(parent)
    rows, unknown_gender = [], 0
    for child in sorted(parents_of):
        ps = sorted(parents_of[child])
        unknown_gender += sum(ds.persons[p].gender is None for p in ps)
        fathers = [p for p in ps if ds.persons[p].gender == "m"]
        mothers = [p for p in ps if ds.persons[p].gender == "f"]
        for f in fathers:
            for m in mothers:
                rows.append(_triplet_row(ds, f, m, child, "matrix"))
    return rows, {"children_with_parents": len(parents_of), "parents_unknown_gender": unknown_gender}


def mine_pseudo_triplets(ds: KinshipDataset, min_age_gap: float = 15,
                         exclude_grandparents: bool = True) -> tuple[list[dict], dict]:
    """Option B: guess (F, M, C) triplets from related pairs + estimated gender/age.

    A triplet (X, Y, C) is accepted when (spec rules):
      1. C is related to both X and Y (same family),
      2. X and Y are NOT related to each other (spouses are not blood relatives),
      3. X is predicted male and Y predicted female,
      4. both X and Y look at least `min_age_gap` years older than C.
    Optional extra rule (`exclude_grandparents`): X is dropped if another relative Q
    of C is related to X and lies one full generation between C and X. That is the
    typical grandparent -> parent -> child chain, and grandparents otherwise pass
    rules 1-4 easily.
    """
    stats = Counter()
    rows = []
    for c, child in sorted(ds.persons.items()):
        if child.age is None:
            stats["children_skipped_no_age"] += 1
            continue
        rels = [r for r in ds.relatives(c) if ds.persons[r].fid == child.fid]
        older = [r for r in rels if ds.persons[r].age is not None
                 and ds.persons[r].gender is not None
                 and ds.persons[r].age >= child.age + min_age_gap]
        if exclude_grandparents:
            kept = []
            for x in older:
                ax = ds.persons[x].age
                middle = [q for q in rels if q != x and ds.is_related(q, x)
                          and ds.persons[q].age is not None
                          and ds.persons[q].age >= child.age + min_age_gap
                          and ax >= ds.persons[q].age + min_age_gap]
                if middle:
                    stats["candidates_removed_as_grandparent"] += 1
                else:
                    kept.append(x)
            older = kept
        males = sorted(x for x in older if ds.persons[x].gender == "m")
        females = sorted(y for y in older if ds.persons[y].gender == "f")
        n_before = len(rows)
        for x in males:
            for y in females:
                if ds.is_related(x, y):
                    stats["rejected_parents_related"] += 1
                    continue
                rows.append(_triplet_row(ds, x, y, c, "pseudo"))
        n_new = len(rows) - n_before
        stats["children_with_triplet"] += n_new > 0
        stats["children_with_multiple_triplets"] += n_new > 1
    stats["pseudo_triplets"] = len(rows)
    return rows, dict(stats)


def _triplet_row(ds: KinshipDataset, f: str, m: str, c: str, source: str,
                 label: int = 1, neg_type: str = "pos") -> dict:
    return {"F": f, "M": m, "C": c, "label": label, "kind": kind_of(ds.persons[c].gender),
            "neg_type": neg_type, "fid": ds.persons[c].fid, "source": source}


def build_positive_triplets(ds: KinshipDataset, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Pick the positive-triplet source that fits the dataset format."""
    if ds.format == "fiw":
        if ds.official_triplets and cfg["data"].get("use_official_triplets", True):
            rows = [_triplet_row(ds, f, m, c, "official") for f, m, c in ds.official_triplets]
            stats = {"source": "official", "official_triplets": len(rows)}
        else:
            rows, stats = triplets_from_parent_child(ds)
            stats["source"] = "matrix"
    else:
        m = cfg["mining"]
        rows, stats = mine_pseudo_triplets(ds, m["min_age_gap"], m.get("exclude_grandparents", True))
        stats["source"] = "pseudo"
    df = pd.DataFrame(rows, columns=TRIPLET_COLUMNS).drop_duplicates(["F", "M", "C"])
    return df.reset_index(drop=True), stats


# ---------------------------------------------------------------------- negatives
class PersonPool:
    """Candidates for negative sampling, with fast numpy filtering."""

    def __init__(self, ds: KinshipDataset, pids):
        self.pids = np.array(sorted(pids), dtype=object)
        self.fids = np.array([ds.persons[p].fid for p in self.pids], dtype=object)
        self.genders = np.array([ds.persons[p].gender or "" for p in self.pids], dtype=object)
        self.ages = np.array([np.nan if ds.persons[p].age is None else ds.persons[p].age
                              for p in self.pids], dtype=float)

    def sample(self, rng: np.random.Generator, *, gender: str | None, age: float | None,
               exclude_fids: set[str], exclude_pids: set[str], k: int) -> str | None:
        """A random stranger: different family, same gender (if known), and one of
        the k closest in age (if ages are known). None if nobody qualifies."""
        if len(self.pids) == 0:
            return None
        mask = ~np.isin(self.fids, list(exclude_fids)) & ~np.isin(self.pids, list(exclude_pids))
        if gender:
            mask &= self.genders == gender
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return None
        if age is not None:
            with_age = idx[~np.isnan(self.ages[idx])]
            if len(with_age):
                order = np.argsort(np.abs(self.ages[with_age] - age), kind="stable")
                idx = with_age[order[:k]]
        return str(self.pids[rng.choice(idx)])


def make_main_negatives(ds: KinshipDataset, pos: pd.DataFrame, pool: PersonPool,
                        rng: np.random.Generator, k: int) -> pd.DataFrame:
    """(F, M, C') for every positive: replace the child with a similar stranger."""
    rows = []
    for src, t in zip(pos.index, pos.itertuples(index=False)):
        child = ds.persons[t.C]
        c2 = pool.sample(rng, gender=child.gender, age=child.age,
                         exclude_fids={t.fid, ds.persons[t.F].fid, ds.persons[t.M].fid},
                         exclude_pids={t.F, t.M, t.C}, k=k)
        if c2 is not None:
            rows.append({**t._asdict(), "C": c2, "label": 0, "neg_type": "main", "_src": src})
    return pd.DataFrame(rows, columns=TRIPLET_COLUMNS + ["_src"])


def make_hard_negatives(ds: KinshipDataset, pos: pd.DataFrame, adults: PersonPool,
                        rng: np.random.Generator, k: int) -> pd.DataFrame:
    """(F', M, C) or (F, M', C): swap ONE parent (chosen at random) for an unrelated
    adult of the same gender and similar age. The child stays the true child."""
    rows = []
    for src, t in zip(pos.index, pos.itertuples(index=False)):
        role = "F" if rng.random() < 0.5 else "M"
        old = getattr(t, role)
        new = adults.sample(rng, gender=ds.persons[old].gender, age=ds.persons[old].age,
                            exclude_fids={t.fid, ds.persons[t.F].fid, ds.persons[t.M].fid},
                            exclude_pids={t.F, t.M, t.C}, k=k)
        if new is not None:
            rows.append({**t._asdict(), role: new, "label": 0, "neg_type": f"hard_{role}", "_src": src})
    return pd.DataFrame(rows, columns=TRIPLET_COLUMNS + ["_src"])


def balance(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    """Positives that received a negative + those negatives: exactly 50% / 50%.

    Each negative remembers the row of the positive it was made from (`_src`), so
    a positive whose negative could not be sampled is dropped together with it.
    """
    kept_pos = pos.loc[neg["_src"].tolist()] if len(neg) else pos.iloc[:0]
    return pd.concat([kept_pos, neg.drop(columns="_src")], ignore_index=True)[TRIPLET_COLUMNS]


# ---------------------------------------------------------------------- pairs
def pair_relation(ds: KinshipDataset, parent: str, child: str) -> str:
    """FS / FD / MS / MD when both genders are known, else 'PC' (parent-child)."""
    pg, cg = ds.persons[parent].gender, ds.persons[child].gender
    if pg and cg:
        return {"m": "F", "f": "M"}[pg] + {"m": "S", "f": "D"}[cg]
    return "PC"


def build_pairs(ds: KinshipDataset, pos_triplets: pd.DataFrame, split_pids: set[str],
                pool: PersonPool, rng: np.random.Generator, k: int) -> pd.DataFrame:
    """Pairwise data for one split: all related pairs + parent-child pairs from the
    triplets (positives), each with one stranger negative from another family."""
    oriented = {(p, c) for p, c in ds.parent_child}
    oriented |= {(t.F, t.C) for t in pos_triplets.itertuples()}
    oriented |= {(t.M, t.C) for t in pos_triplets.itertuples()}
    keys = {k_ for k_ in ds.related_pairs if k_[0] in split_pids and k_[1] in split_pids}
    keys |= {pair_key(p, c) for p, c in oriented if p in split_pids and c in split_pids}

    pos_rows = []
    for a, b in sorted(keys):
        if (b, a) in oriented:
            a, b = b, a
        rel = pair_relation(ds, a, b) if (a, b) in oriented else "related"
        pos_rows.append({"p1": a, "p2": b, "label": 1, "rel": rel, "fid": ds.persons[a].fid})

    neg_rows = []
    for r in pos_rows:
        other = ds.persons[r["p2"]]
        stranger = pool.sample(rng, gender=other.gender, age=other.age,
                               exclude_fids={ds.persons[r["p1"]].fid, other.fid},
                               exclude_pids={r["p1"], r["p2"]}, k=k)
        if stranger is not None:
            neg_rows.append({**r, "p2": stranger, "label": 0})
    n = min(len(pos_rows), len(neg_rows))
    return pd.DataFrame(pos_rows[:n] + neg_rows[:n], columns=PAIR_COLUMNS)


# ---------------------------------------------------------------------- everything
def build_all_splits(ds: KinshipDataset, cfg: dict) -> dict:
    """Family-disjoint splits + positives + negatives + pairs for every split.

    Returns {"families": {split: [fid]}, "triplets": {split: df},
             "hard": {split: df}, "pairs": {split: df}, "stats": {...}}.
    """
    seed = int(cfg["seed"])
    k = int(cfg["negatives"]["age_neighbors"])
    adult_age = float(cfg["negatives"]["adult_min_age"])
    hard_splits = set(cfg["negatives"].get("hard_splits", ["val", "test"]))

    positives, pos_stats = build_positive_triplets(ds, cfg)
    fams_with_triplets = set(positives["fid"])
    strata = {fid: ("has_triplets" if fid in fams_with_triplets else "no_triplets")
              for fid in ds.families}
    ratios = {s: float(cfg["splits"][s]) for s in SPLITS}
    families = split_families(list(ds.families), ratios, seed, strata)

    out = {"families": families, "triplets": {}, "hard": {}, "pairs": {},
           "stats": {"format": ds.format, "positives": pos_stats, "splits": {}}}
    for i, split in enumerate(SPLITS):
        fam_set = set(families[split])
        split_pids = {p for p, person in ds.persons.items() if person.fid in fam_set}
        in_split = positives[[all(ds.persons[x].fid in fam_set for x in (t.F, t.M, t.C))
                              for t in positives.itertuples()]].reset_index(drop=True)

        # Separate, reproducible random streams per split and per purpose.
        rng_main = np.random.default_rng([seed, i, 1])
        rng_hard = np.random.default_rng([seed, i, 2])
        rng_pair = np.random.default_rng([seed, i, 3])
        rng_order = np.random.default_rng([seed, i, 4])

        # Shuffle positives once so that, if a few get no negative, the dropped ones are random.
        in_split = in_split.sample(frac=1.0, random_state=rng_order).reset_index(drop=True)
        pool = PersonPool(ds, split_pids)
        neg = make_main_negatives(ds, in_split, pool, rng_main, k)
        out["triplets"][split] = balance(in_split, neg)

        if split in hard_splits:
            adults = {x for t in in_split.itertuples() for x in (t.F, t.M)}
            adults |= {p for p in split_pids if ds.persons[p].gender
                       and ds.persons[p].age is not None and ds.persons[p].age >= adult_age}
            hard = make_hard_negatives(ds, in_split, PersonPool(ds, adults), rng_hard, k)
            out["hard"][split] = balance(in_split, hard)

        out["pairs"][split] = build_pairs(ds, in_split, split_pids, pool, rng_pair, k)
        out["stats"]["splits"][split] = {
            "families": len(fam_set), "persons": len(split_pids),
            "positive_triplets": int(in_split.shape[0]),
            "triplets_after_balance": int(len(out["triplets"][split])),
            "kinds": out["triplets"][split].query("label == 1")["kind"].value_counts().to_dict(),
            "hard_triplets": int(len(out["hard"].get(split, []))),
            "pairs": int(len(out["pairs"][split])),
        }
    return out

