"""Format detection and loading of both dataset layouts."""
from kinship.data import detect_format


def test_detect_format(kaggle_synth, fiw_synth):
    assert detect_format(kaggle_synth["root"]) == "kaggle"
    assert detect_format(fiw_synth["root"]) == "fiw"


def test_kaggle_loading(kaggle_synth):
    ds, truth = kaggle_synth["ds"], kaggle_synth["truth"]
    assert ds.format == "kaggle"
    assert set(ds.persons) == set(truth["persons"])
    assert ds.stats["pairs_dropped_missing_person"] == 0
    assert all(p.images for p in ds.persons.values())
    # Spouses are not blood relatives, so they must not be listed as related.
    for f, m, _ in truth["triplets"]:
        assert not ds.is_related(f, m)


def test_fiw_matrix_parsing(fiw_synth):
    ds, truth = fiw_synth["ds"], fiw_synth["truth"]
    assert ds.format == "fiw"
    assert ds.stats["matrix_inconsistent_cells"] == 0
    # Gender comes from mid.csv (ground truth), not from the estimate.
    for pid, info in truth["persons"].items():
        assert ds.persons[pid].gender == info["gender"]
    # Every true (father, child) and (mother, child) edge is read in the right direction.
    for f, m, c in truth["triplets"]:
        assert (f, c) in ds.parent_child and (m, c) in ds.parent_child
        assert (c, f) not in ds.parent_child
        assert tuple(sorted((f, m))) in ds.spouses


def test_find_data_root_without_full_walk(fiw_synth, kaggle_synth):
    from kinship.data import find_data_root, scan_tree

    families, csvs = scan_tree(str(fiw_synth["root"]))
    assert len(families) == 40 and all((f / "mid.csv").exists() for f in families)
    # The base folder holds several attached datasets, like /kaggle/input.
    base = fiw_synth["root"].parent
    assert find_data_root(base) == families[0].parent
    assert find_data_root(kaggle_synth["root"]) == kaggle_synth["root"]
