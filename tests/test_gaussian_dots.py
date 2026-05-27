import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dots import Dot, counts_in_crop, gaussian_counts_in_crop


def test_gaussian_counts_in_crop_softens_nearby_dots():
    dots = [Dot(x=310, y=310, class_idx=0)]
    hard = counts_in_crop(dots, 0, 0, 299)
    soft = gaussian_counts_in_crop(dots, 0, 0, 299, sigma=25.0)
    assert hard[0] == 0.0
    assert soft[0] > 0.0
    assert soft[0] < 1.0


def test_gaussian_counts_in_crop_matches_hard_inside_crop():
    dots = [Dot(x=50, y=50, class_idx=2)]
    hard = counts_in_crop(dots, 0, 0, 299)
    soft = gaussian_counts_in_crop(dots, 0, 0, 299, sigma=25.0)
    assert hard[2] == 1.0
    assert abs(soft[2] - 1.0) < 0.01
