import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.tiling import TileWindow, aggregate_tile_predictions, iter_tile_windows, tile_area_fraction


def test_iter_tile_windows_covers_image():
    wins = list(iter_tile_windows(500, 400, tile_size=128, stride=128, shifts=1))
    assert len(wins) > 0
    for w in wins:
        assert w.x1 - w.x0 <= 128
        assert w.y1 - w.y0 <= 128


def test_iter_tile_windows_unique_keys():
    wins = list(iter_tile_windows(640, 480, tile_size=299, stride=149, shifts=5))
    keys = [w.key() for w in wins]
    assert len(keys) == len(set(keys)), "duplicate tile windows detected"


def test_aggregate_area_weighted():
    wins = [
        TileWindow(0, 0, 100, 100, 0),
        TileWindow(100, 0, 200, 100, 0),
    ]
    preds = [np.ones(5), np.ones(5) * 2]
    out = aggregate_tile_predictions(preds, wins, 200, 100, mode="area_weighted")
    assert out.shape == (5,)
    assert np.all(out >= 0)


def test_aggregate_sum():
    wins = [TileWindow(0, 0, 50, 50, 0), TileWindow(50, 0, 100, 50, 0)]
    preds = [np.array([1, 0, 0, 0, 0]), np.array([2, 0, 0, 0, 0])]
    out = aggregate_tile_predictions(preds, wins, 100, 50, mode="sum")
    assert out[0] == 3.0


def test_aggregate_length_mismatch_raises():
    wins = [TileWindow(0, 0, 50, 50, 0)]
    try:
        aggregate_tile_predictions([np.zeros(5), np.zeros(5)], wins, 100, 50, mode="sum")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_tile_area_fraction():
    w = TileWindow(0, 0, 100, 100, 0)
    assert abs(tile_area_fraction(w, 200, 200) - 0.25) < 1e-6
