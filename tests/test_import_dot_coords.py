import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.import_dot_coords import lopuhin_to_dot_labels


def test_lopuhin_to_dot_labels_maps_columns_and_classes():
    src = pd.DataFrame(
        {
            "tid": [0, 0, 1],
            "cls": [0, 4, 2],
            "row": [10, 20, 30],
            "col": [100, 200, 300],
        }
    )
    out = lopuhin_to_dot_labels(src)
    assert list(out.columns) == ["image_id", "x", "y", "class"]
    assert out.iloc[0].tolist() == ["0", 100, 10, "adult_males"]
    assert out.iloc[1].tolist() == ["0", 200, 20, "pups"]
    assert out.iloc[2].tolist() == ["1", 300, 30, "adult_females"]
