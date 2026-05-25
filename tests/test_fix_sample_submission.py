import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.fix_sample_submission import build_sample
from data.targets import COUNT_COLUMNS


def test_build_sample_from_test_dir(mini_data_dir):
    df = build_sample(mini_data_dir, test_subdir="Test", limit=0)
    assert len(df) == 2
    assert list(df.columns) == ["id", *COUNT_COLUMNS]
    assert df["id"].tolist() == ["test_000.jpg", "test_001.jpg"]


def test_build_sample_respects_limit(mini_data_dir):
    df = build_sample(mini_data_dir, test_subdir="Test", limit=1)
    assert len(df) == 1
