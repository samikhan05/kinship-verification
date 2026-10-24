"""Triplet construction (both options), mining rules and negatives."""
import copy

import numpy as np
import pandas as pd

from kinship.data import Person, KinshipDataset, pair_key
from kinship.triplets import (build_all_splits, build_positive_triplets, mine_pseudo_triplets,
                              triplets_from_parent_child)


def _set(df_or_rows):
    rows = df_or_rows.to_dict("records") if isinstance(df_or_rows, pd.DataFrame) else df_or_rows
    return {(r["F"], r["M"], r["C"]) for r in rows}


def test_option_a_triplets_match_truth(fiw_synth):
    rows, _ = triplets_from_parent_child(fiw_synth["ds"])
    assert _set(rows) == set(fiw_synth["truth"]["triplets"])


def test_option_b_mining_recovers_truth_with_gt_attributes(kaggle_synth):
    rows, stats = mine_pseudo_triplets(kaggle_synth["ds"], 15, exclude_grandparents=True)
    assert _set(rows) == set(kaggle_synth["truth"]["triplets"])
    assert stats["pseudo_triplets"] == len(rows)


def test_without_grandparent_filter_grandparents_leak_in(kaggle_synth):
    """Shows why the extra filter exists: rules 1-4 alone accept grandparents."""
    rows, _ = mine_pseudo_triplets(kaggle_synth["ds"], 15, exclude_grandparents=False)
    truth = set(kaggle_synth["truth"]["triplets"])
    assert truth <= _set(rows)
    assert len(_set(rows) - truth) > 0


def _tiny(gender_x="m", gender_y="f", age_x=40, age_y=40, x_related_y=False):
    """One family: X, Y and child C (age 10), C related to both."""
    persons = {pid: Person(pid, "F0001", ["a.jpg"], g, a) for pid, g, a in
               [("F0001/MID1", gender_x, age_x), ("F0001/MID2", gender_y, age_y),
                ("F0001/MID3", "m", 10)]}
    pairs = {pair_key("F0001/MID1", "F0001/MID3"), pair_key("F0001/MID2", "F0001/MID3")}
    if x_related_y:
        pairs.add(pair_key("F0001/MID1", "F0001/MID2"))
    return KinshipDataset("kaggle", None, persons, pairs)


def test_mining_rules():
    ok, _ = mine_pseudo_triplets(_tiny(), 15)
    assert _set(ok) == {("F0001/MID1", "F0001/MID2", "F0001/MID3")}
    assert ok[0]["kind"] == "FMS"
    assert not mine_pseudo_triplets(_tiny(x_related_y=True), 15)[0]     # parents related
    assert not mine_pseudo_triplets(_tiny(gender_y="m"), 15)[0]         # two males
    assert not mine_pseudo_triplets(_tiny(age_y=20), 15)[0]             # only 10 years older


def test_splits_balanced_and_negatives_valid(kaggle_synth):
    ds, cfg = kaggle_synth["ds"], kaggle_synth["cfg"]
    out = build_all_splits(ds, cfg)
    for split, df in out["triplets"].items():
        assert len(df) > 0
        assert (df["label"] == 1).sum() == (df["label"] == 0).sum()   # 50 / 50
        for t in df.query("label == 0").itertuples():
            # main negative: C' is a stranger with the same gender as the real child
            assert ds.persons[t.C].fid != ds.persons[t.F].fid
            assert ds.persons[t.C].gender == ("m" if t.kind == "FMS" else "f")
    for split, df in out["hard"].items():
        assert (df["label"] == 1).sum() == (df["label"] == 0).sum()
        for t in df.query("label == 0").itertuples():
            swapped = t.F if t.neg_type == "hard_F" else t.M
            assert ds.persons[swapped].fid != ds.persons[t.C].fid
            assert ds.persons[swapped].gender == ("m" if t.neg_type == "hard_F" else "f")
    for split, df in out["pairs"].items():
        assert (df["label"] == 1).sum() == (df["label"] == 0).sum()
        neg = df.query("label == 0")
        assert all(ds.persons[a].fid != ds.persons[b].fid for a, b in zip(neg.p1, neg.p2))


def test_build_is_deterministic(kaggle_synth):
    a = build_all_splits(kaggle_synth["ds"], kaggle_synth["cfg"])
    b = build_all_splits(kaggle_synth["ds"], kaggle_synth["cfg"])
    for split in a["triplets"]:
        pd.testing.assert_frame_equal(a["triplets"][split], b["triplets"][split])
        pd.testing.assert_frame_equal(a["pairs"][split], b["pairs"][split])


def test_option_a_pairs_have_roles(fiw_synth):
    out = build_all_splits(fiw_synth["ds"], fiw_synth["cfg"])
    rels = set(pd.concat(out["pairs"].values())["rel"])
    assert {"FS", "FD", "MS", "MD"} <= rels
