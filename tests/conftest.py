import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.targets import COUNT_COLUMNS, SUBMISSION_ID_COL, pred_vector_to_submission_row


@pytest.fixture
def mini_data_dir(tmp_path):
    """Minimal layout matching Kaggle structure for unit tests."""
    d = tmp_path / "datasets"
    (d / "Train").mkdir(parents=True)
    (d / "TrainDotted").mkdir()
    (d / "Test").mkdir()

    rows = []
    for i in range(4):
        name = f"img_{i:03d}.jpg"
        img = Image.new("RGB", (400, 300), color=(i * 40, 100, 150))
        img.save(d / "Train" / name)
        img.save(d / "TrainDotted" / name)
        rows.append({
            "id": Path(name).stem,
            "adult_males": float(i),
            "subadult_males": 0.5,
            "adult_females": 1.0,
            "juveniles": 0.0,
            "pups": 2.0,
        })

    for j in range(2):
        name = f"test_{j:03d}.jpg"
        Image.new("RGB", (512, 400), color=(200, 200, 200)).save(d / "Test" / name)

    pd.DataFrame(rows).to_csv(d / "train.csv", index=False)
    sub_rows = []
    for j in range(2):
        row = {SUBMISSION_ID_COL: f"test_{j:03d}"}
        for col in COUNT_COLUMNS:
            row[col] = 0
        sub_rows.append(row)
    pd.DataFrame(sub_rows).to_csv(d / "sample_submission.csv", index=False)
    (d / "MismatchedTrainImages.txt").write_text("")

    return d
