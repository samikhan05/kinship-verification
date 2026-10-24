"""Shared pytest fixtures: synthetic datasets in both formats + a matching config."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kinship.data import apply_attributes, load_dataset  # noqa: E402
from kinship.synthetic import make_synthetic  # noqa: E402
from kinship.utils import load_config  # noqa: E402


def _setup(tmp_path_factory, fmt):
    root = tmp_path_factory.mktemp(f"synth_{fmt}")
    truth = make_synthetic(root, fmt, n_families=40, seed=0)
    cfg = load_config(overrides=[f"paths.data_root={root.as_posix()}",
                                 f"paths.work_dir={(root / 'work').as_posix()}",
                                 f"paths.output_dir={(root / 'out').as_posix()}"])
    ds = load_dataset(cfg)
    apply_attributes(ds, pd.read_csv(root / "attributes_gt.csv"))
    return {"root": root, "truth": truth, "cfg": cfg, "ds": ds}


@pytest.fixture(scope="session")
def kaggle_synth(tmp_path_factory):
    return _setup(tmp_path_factory, "kaggle")


@pytest.fixture(scope="session")
def fiw_synth(tmp_path_factory):
    return _setup(tmp_path_factory, "fiw")
