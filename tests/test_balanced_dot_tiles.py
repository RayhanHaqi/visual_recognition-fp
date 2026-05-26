import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import SeaLionTileDataset
from data.dots import CLASS_TO_IDX, Dot
from data.targets import load_train_counts


def test_balanced_dots_positive_tile_counts_known_dot(mini_data_dir):
    counts_df = load_train_counts(mini_data_dir, exclude_mismatched=False)
    path = sorted((mini_data_dir / "Train").glob("*.jpg"))[0]
    dots_by_image = {
        path.stem: [
            Dot(50, 50, CLASS_TO_IDX["adult_males"]),
        ]
    }
    ds = SeaLionTileDataset(
        [path],
        counts_df,
        tile_size=128,
        tiles_per_image=1,
        train=False,
        label_mode="balanced_dots",
        dots_by_image=dots_by_image,
        balanced_positive_fraction=1.0,
    )

    tensor, target, img_id = ds[0]

    assert img_id == path.stem
    assert tensor.shape == (3, 128, 128)
    assert target.shape == (5,)
    assert float(target[CLASS_TO_IDX["adult_males"]]) == 1.0


def test_balanced_dots_background_tile_returns_zero_target(mini_data_dir):
    counts_df = load_train_counts(mini_data_dir, exclude_mismatched=False)
    path = sorted((mini_data_dir / "Train").glob("*.jpg"))[0]
    dots_by_image = {path.stem: [Dot(10, 10, CLASS_TO_IDX["adult_males"])]}
    ds = SeaLionTileDataset(
        [path],
        counts_df,
        tile_size=64,
        tiles_per_image=1,
        train=False,
        label_mode="balanced_dots",
        dots_by_image=dots_by_image,
        balanced_positive_fraction=0.0,
    )

    _, target, _ = ds[0]

    assert torch.equal(target, torch.zeros(5))


def test_balanced_dots_preserves_existing_area_and_dots_modes(mini_data_dir):
    counts_df = load_train_counts(mini_data_dir, exclude_mismatched=False)
    path = sorted((mini_data_dir / "Train").glob("*.jpg"))[0]
    dots_by_image = {path.stem: [Dot(160, 100, CLASS_TO_IDX["pups"])]}

    area_ds = SeaLionTileDataset(
        [path],
        counts_df,
        tile_size=128,
        tiles_per_image=1,
        train=False,
        label_mode="area",
        dots_by_image=dots_by_image,
    )
    dots_ds = SeaLionTileDataset(
        [path],
        counts_df,
        tile_size=128,
        tiles_per_image=1,
        train=False,
        label_mode="dots",
        dots_by_image=dots_by_image,
    )

    _, area_target, _ = area_ds[0]
    _, dots_target, _ = dots_ds[0]

    assert float(area_target.sum()) > 0.0
    assert float(dots_target[CLASS_TO_IDX["pups"]]) == 1.0
