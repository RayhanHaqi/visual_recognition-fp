import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.metrics import rmse_numpy, rmse_per_class


def test_rmse_zero():
    a = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    assert rmse_numpy(a, a) == 0.0


def test_rmse_known():
    pred = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)
    tgt = np.array([[3, 4, 0, 0, 0]], dtype=np.float32)
    # mse = (9+16)/5 = 5, rmse = sqrt(5)
    assert abs(rmse_numpy(pred, tgt) - np.sqrt(5)) < 1e-5


def test_rmse_per_class():
    pred = np.zeros((2, 5))
    tgt = np.ones((2, 5))
    d = rmse_per_class(pred, tgt)
    assert len(d) == 5
    assert all(abs(v - 1.0) < 1e-5 for v in d.values())
