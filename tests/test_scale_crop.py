"""Source-crop scale augmentation counts dots in the pre-resize window."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import SeaLionTileDataset
from data.dots import Dot


def test_scale_crop_larger_source_includes_extra_dot(monkeypatch, mini_data_dir):
    counts_df = __import__("pandas").read_csv(mini_data_dir / "train.csv")
    train_paths = sorted((mini_data_dir / "Train").glob("*.jpg"))
    dots = {
        train_paths[0].stem: [
            Dot(50, 50, 0),
            Dot(200, 200, 0),
        ]
    }
    ds = SeaLionTileDataset(
        train_paths[:1],
        counts_df,
        tile_size=128,
        tiles_per_image=1,
        train=True,
        label_mode="balanced_dots",
        dots_by_image=dots,
        balanced_positive_fraction=1.0,
        scale_min=0.49,
        scale_max=0.51,
    )
    monkeypatch.setattr(np.random, "uniform", lambda *_: 0.5)

    tensor, target, _ = ds[0]
    assert tensor.shape == (3, 128, 128)
    assert target[0].item() >= 1.0


def test_scale_crop_disabled_uses_tile_size_only(mini_data_dir):
    counts_df = __import__("pandas").read_csv(mini_data_dir / "train.csv")
    train_paths = sorted((mini_data_dir / "Train").glob("*.jpg"))
    ds = SeaLionTileDataset(
        train_paths[:1],
        counts_df,
        tile_size=64,
        tiles_per_image=1,
        train=True,
        label_mode="area",
    )
    assert ds._sample_source_size(500, 500) == 64


def test_scale_crop_allows_zoom_in_source_smaller_than_tile(monkeypatch, mini_data_dir):
    counts_df = __import__("pandas").read_csv(mini_data_dir / "train.csv")
    train_paths = sorted((mini_data_dir / "Train").glob("*.jpg"))
    ds = SeaLionTileDataset(
        train_paths[:1],
        counts_df,
        tile_size=100,
        tiles_per_image=1,
        train=True,
        label_mode="balanced_dots",
        dots_by_image={train_paths[0].stem: [Dot(50, 50, 0)]},
        scale_min=0.8,
        scale_max=1.25,
    )
    monkeypatch.setattr(np.random, "uniform", lambda *_: 1.25)

    assert ds._sample_source_size(500, 500) == 80
