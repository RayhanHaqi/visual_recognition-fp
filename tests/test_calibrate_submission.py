import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.submission_ops import blend_submissions, calibrate_submission, load_submission
from data.targets import COUNT_COLUMNS, SUBMISSION_COLUMNS, SUBMISSION_ID_COL


def _submission(rows):
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def test_calibrate_submission_scales_selected_classes_and_finalizes():
    df = _submission(
        [
            {
                SUBMISSION_ID_COL: "0.jpg",
                "adult_males": 1.2,
                "subadult_males": 2.0,
                "adult_females": 3.0,
                "juveniles": 4.0,
                "pups": 5.0,
            }
        ]
    )

    out = calibrate_submission(df, {"pups": 1.2, "adult_males": 0.5})

    assert list(out.columns) == SUBMISSION_COLUMNS
    assert out.loc[0, SUBMISSION_ID_COL] == 0
    assert out.loc[0, "adult_males"] == 1
    assert out.loc[0, "pups"] == 6
    assert out.loc[0, "adult_females"] == 3


def test_blend_submissions_averages_counts_by_test_id():
    left = _submission(
        [
            {SUBMISSION_ID_COL: 0, **{col: 10 for col in COUNT_COLUMNS}},
            {SUBMISSION_ID_COL: 1, **{col: 20 for col in COUNT_COLUMNS}},
        ]
    )
    right = _submission(
        [
            {SUBMISSION_ID_COL: 1, **{col: 40 for col in COUNT_COLUMNS}},
            {SUBMISSION_ID_COL: 0, **{col: 30 for col in COUNT_COLUMNS}},
        ]
    )

    out = blend_submissions([left, right], weights=[0.25, 0.75])

    assert out[SUBMISSION_ID_COL].tolist() == [0, 1]
    assert out.loc[0, "pups"] == 25
    assert out.loc[1, "adult_females"] == 35


def test_calibrate_submission_rejects_unknown_scale_column():
    df = _submission([{SUBMISSION_ID_COL: 0, **{col: 1 for col in COUNT_COLUMNS}}])

    with pytest.raises(ValueError, match="Unknown count column"):
        calibrate_submission(df, {"not_a_class": 1.2})


def test_load_submission_rejects_duplicate_test_ids():
    df = _submission(
        [
            {SUBMISSION_ID_COL: 0, **{col: 1 for col in COUNT_COLUMNS}},
            {SUBMISSION_ID_COL: 0, **{col: 2 for col in COUNT_COLUMNS}},
        ]
    )

    with pytest.raises(ValueError, match="Duplicate test_id"):
        load_submission(df)


def test_blend_submissions_rejects_mismatched_test_ids():
    left = _submission([{SUBMISSION_ID_COL: 0, **{col: 1 for col in COUNT_COLUMNS}}])
    right = _submission([{SUBMISSION_ID_COL: 1, **{col: 2 for col in COUNT_COLUMNS}}])

    with pytest.raises(ValueError, match="same test_id values"):
        blend_submissions([left, right])


def test_blend_submissions_rejects_wrong_weight_count():
    left = _submission([{SUBMISSION_ID_COL: 0, **{col: 1 for col in COUNT_COLUMNS}}])
    right = _submission([{SUBMISSION_ID_COL: 0, **{col: 2 for col in COUNT_COLUMNS}}])

    with pytest.raises(ValueError, match="weights length"):
        blend_submissions([left, right], weights=[0.5])
