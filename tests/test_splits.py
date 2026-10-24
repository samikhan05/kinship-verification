"""No family may appear in more than one split - anywhere in the prepared data."""
import pytest

from kinship.data import split_families
from kinship.triplets import build_all_splits


def test_split_families_disjoint_and_complete():
    fids = [f"F{i:04d}" for i in range(200)]
    s = split_families(fids, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    sets = [set(v) for v in s.values()]
    assert set.union(*sets) == set(fids)
    assert sum(len(x) for x in sets) == len(fids)              # no duplicates
    assert len(s["train"]) == 140 and len(s["val"]) == 30 and len(s["test"]) == 30
    assert s == split_families(fids, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    assert s != split_families(fids, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)


@pytest.mark.parametrize("which", ["kaggle_synth", "fiw_synth"])
def test_no_family_leakage(which, request):
    data = request.getfixturevalue(which)
    ds = data["ds"]
    out = build_all_splits(ds, data["cfg"])
    fams = {s: set(v) for s, v in out["families"].items()}
    assert not (fams["train"] & fams["val"])
    assert not (fams["train"] & fams["test"])
    assert not (fams["val"] & fams["test"])

    # Every person used in a split (positives AND negatives) belongs to that split's families.
    for split in fams:
        frames = [out["triplets"][split], out["pairs"][split]]
        if split in out["hard"]:
            frames.append(out["hard"][split])
        for df in frames:
            for col in [c for c in ("F", "M", "C", "p1", "p2") if c in df]:
                assert set(df[col].map(lambda p: ds.persons[p].fid)) <= fams[split], (split, col)
