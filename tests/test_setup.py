import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from setup import (
    MIN_FREE_GB,
    check_dataset_exists,
    check_disk_space,
    dataset_summary,
)


def test_check_dataset_mini(mini_data_dir):
    assert check_dataset_exists(mini_data_dir) is True
    s = dataset_summary(mini_data_dir)
    assert s["ok"] is True
    assert s["n_train"] == 4
    assert s["n_test"] == 2


def test_check_dataset_missing(tmp_path):
    assert check_dataset_exists(tmp_path / "empty") is False


def test_disk_space_reports_gb():
    ok, free_gb = check_disk_space(ROOT)
    assert free_gb > 0
    assert isinstance(ok, bool)


def test_min_free_gb_sane():
    assert MIN_FREE_GB >= 100
