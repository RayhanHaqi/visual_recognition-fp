import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dot_cache_analysis import (
    analyze_dot_cache,
    class_summary_from_rows,
    dot_cache_gate_failures,
    write_analysis,
)


def _write_train_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_dot_cache(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_class_summary_from_rows_totals_by_class():
    rows = pd.DataFrame(
        [
            {
                "train_adult_males": 2,
                "dot_adult_males": 1,
                "train_pups": 3,
                "dot_pups": 4,
            },
            {
                "train_adult_males": 0,
                "dot_adult_males": 2,
                "train_pups": 1,
                "dot_pups": 1,
            },
        ]
    )
    summary = class_summary_from_rows(rows)
    assert summary.loc["adult_males", "train_total"] == 2
    assert summary.loc["adult_males", "dot_total"] == 3
    assert summary.loc["pups", "train_total"] == 4
    assert summary.loc["pups", "dot_total"] == 5


def test_analyze_dot_cache_flags_missing_dotted_and_mismatch(tmp_path):
    data_dir = tmp_path / "datasets"
    (data_dir / "Train").mkdir(parents=True)
    (data_dir / "TrainDotted").mkdir()
    (data_dir / "Train" / "good.jpg").write_bytes(b"x")
    (data_dir / "TrainDotted" / "good.jpg").write_bytes(b"x")
    (data_dir / "Train" / "missing_dotted.jpg").write_bytes(b"x")
    (data_dir / "Train" / "mismatch.jpg").write_bytes(b"x")
    (data_dir / "TrainDotted" / "mismatch.jpg").write_bytes(b"x")

    _write_train_csv(
        data_dir / "train.csv",
        [
            {
                "id": "good",
                "adult_males": 1,
                "subadult_males": 0,
                "adult_females": 0,
                "juveniles": 0,
                "pups": 1,
            },
            {
                "id": "missing_dotted",
                "adult_males": 5,
                "subadult_males": 0,
                "adult_females": 0,
                "juveniles": 0,
                "pups": 0,
            },
            {
                "id": "mismatch",
                "adult_males": 10,
                "subadult_males": 0,
                "adult_females": 0,
                "juveniles": 0,
                "pups": 0,
            },
        ],
    )
    dot_cache = data_dir / "dot_labels.csv"
    _write_dot_cache(
        dot_cache,
        [
            {"image_id": "good", "x": 1, "y": 2, "class": "adult_males"},
            {"image_id": "good", "x": 3, "y": 4, "class": "pups"},
            {"image_id": "mismatch", "x": 5, "y": 6, "class": "pups"},
        ],
    )

    result = analyze_dot_cache(data_dir, dot_cache, rel_err_warn=0.1, abs_err_warn=1.0)
    by_id = result.per_image.set_index("image_id")

    assert bool(by_id.loc["good", "has_dotted"])
    assert by_id.loc["good", "n_dots"] == 2
    assert by_id.loc["good", "total_abs_err"] == 0

    assert not bool(by_id.loc["missing_dotted", "has_dotted"])
    assert "missing_dotted" in result.warnings["missing_dotted"]

    assert by_id.loc["mismatch", "total_abs_err"] == 9
    assert "mismatch" in result.warnings["large_mismatch"]


def test_write_analysis_writes_per_image_and_summary(tmp_path):
    data_dir = tmp_path / "datasets"
    (data_dir / "Train").mkdir(parents=True)
    (data_dir / "TrainDotted").mkdir()
    (data_dir / "Train" / "img.jpg").write_bytes(b"x")
    (data_dir / "TrainDotted" / "img.jpg").write_bytes(b"x")
    _write_train_csv(
        data_dir / "train.csv",
        [
            {
                "id": "img",
                "adult_males": 1,
                "subadult_males": 0,
                "adult_females": 0,
                "juveniles": 0,
                "pups": 0,
            }
        ],
    )
    dot_cache = data_dir / "dot_labels.csv"
    _write_dot_cache(
        dot_cache,
        [{"image_id": "img", "x": 1, "y": 1, "class": "adult_males"}],
    )

    out = tmp_path / "log" / "dot_cache_analysis.csv"
    result = analyze_dot_cache(data_dir, dot_cache)
    write_analysis(result, out)

    assert out.is_file()
    assert out.with_name("dot_cache_analysis_summary.csv").is_file()
    per_image = pd.read_csv(out)
    assert len(per_image) == 1
    assert per_image.loc[0, "image_id"] == "img"


def test_dot_cache_gate_reports_class_relative_error(tmp_path):
    data_dir = tmp_path / "datasets"
    (data_dir / "Train").mkdir(parents=True)
    (data_dir / "TrainDotted").mkdir()
    (data_dir / "Train" / "img.jpg").write_bytes(b"x")
    (data_dir / "TrainDotted" / "img.jpg").write_bytes(b"x")
    _write_train_csv(
        data_dir / "train.csv",
        [
            {
                "id": "img",
                "adult_males": 10,
                "subadult_males": 0,
                "adult_females": 0,
                "juveniles": 0,
                "pups": 0,
            }
        ],
    )
    dot_cache = data_dir / "dot_labels.csv"
    _write_dot_cache(
        dot_cache,
        [{"image_id": "img", "x": 1, "y": 1, "class": "adult_males"}],
    )

    result = analyze_dot_cache(data_dir, dot_cache)
    failures = dot_cache_gate_failures(result, max_class_rel_err=0.10)

    assert failures == ["adult_males rel_err 0.900 > 0.100"]
