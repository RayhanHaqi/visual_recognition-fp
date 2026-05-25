import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.convert_submission import convert_submission
from data.targets import COUNT_COLUMNS, SUBMISSION_COLUMNS, SUBMISSION_ID_COL


def test_convert_legacy_submission():
    legacy = pd.DataFrame(
        {
            "id": ["0.jpg", "1.jpg"],
            "adult_males": [1.0, 2.0],
            "adult_females": [3.0, 4.0],
            "subadult_males": [5.0, 6.0],
            "subadult_females": [7.0, 8.0],
            "pups": [9.0, 10.0],
        }
    )
    out = convert_submission(legacy)
    assert list(out.columns) == SUBMISSION_COLUMNS
    assert out[SUBMISSION_ID_COL].tolist() == [0, 1]
    assert out.loc[0, "juveniles"] == 7.0
    assert out.loc[0, "adult_females"] == 3.0
    assert out.loc[0, "subadult_males"] == 5.0
