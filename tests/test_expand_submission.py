import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.expand_submission import expand_submission
from data.targets import COUNT_COLUMNS, SUBMISSION_ID_COL


def test_expand_submission_fills_missing_with_zero():
    partial = pd.DataFrame(
        {
            SUBMISSION_ID_COL: [0, 1],
            **{col: [1, 2] for col in COUNT_COLUMNS},
        }
    )
    template = pd.DataFrame(
        {
            SUBMISSION_ID_COL: [0, 1, 2],
            **{col: [0, 0, 0] for col in COUNT_COLUMNS},
        }
    )
    out = expand_submission(partial, template)
    assert len(out) == 3
    assert out.loc[out[SUBMISSION_ID_COL] == 2, "juveniles"].iloc[0] == 0
    assert out.loc[out[SUBMISSION_ID_COL] == 0, "adult_males"].iloc[0] == 1
