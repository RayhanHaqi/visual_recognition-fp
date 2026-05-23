import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.targets import COUNT_COLUMNS, load_train_counts


def test_train_counts_load(mini_data_dir):
    df = load_train_counts(mini_data_dir, exclude_mismatched=False)
    assert len(df) == 4
    for c in COUNT_COLUMNS:
        assert c in df.columns


def test_submission_columns_match_sample(mini_data_dir):
    sample = pd.read_csv(mini_data_dir / "sample_submission.csv")
    id_col = "id" if "id" in sample.columns else sample.columns[0]
    assert id_col in sample.columns
    for c in COUNT_COLUMNS:
        assert c in sample.columns


def test_submission_non_negative(mini_data_dir, tmp_path):
    sample = pd.read_csv(mini_data_dir / "sample_submission.csv")
    out = tmp_path / "sub.csv"
    sample.copy().to_csv(out, index=False)
    df = pd.read_csv(out)
    for c in COUNT_COLUMNS:
        assert (df[c] >= 0).all()


def test_setup_check_dataset(mini_data_dir):
    from setup import check_dataset_exists

    assert check_dataset_exists(mini_data_dir) is True
    assert check_dataset_exists(mini_data_dir / "missing") is False
