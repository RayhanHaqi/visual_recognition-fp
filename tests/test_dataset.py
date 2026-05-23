import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import SeaLionImageDataset, SeaLionTileDataset, build_train_val_paths
from data.targets import list_train_images, load_train_counts


def test_tile_dataset_item(mini_data_dir):
    paths = list_train_images(mini_data_dir, exclude_mismatched=False)
    df = load_train_counts(mini_data_dir, exclude_mismatched=False)
    ds = SeaLionTileDataset(paths[:2], df, tile_size=128, tiles_per_image=4, train=True)
    x, y, img_id = ds[0]
    assert x.shape == (3, 128, 128)
    assert y.shape == (5,)
    assert isinstance(img_id, str)


def test_image_dataset_item(mini_data_dir):
    paths = list_train_images(mini_data_dir, exclude_mismatched=False)
    df = load_train_counts(mini_data_dir, exclude_mismatched=False)
    ds = SeaLionImageDataset(paths[:1], df, tile_size=128, train=False)
    x, y, _ = ds[0]
    assert y.shape == (5,)


def test_train_val_split(mini_data_dir):
    train_p, val_p, _ = build_train_val_paths(mini_data_dir, val_frac=0.25, seed=0)
    assert len(train_p) + len(val_p) == 4
    assert len(val_p) >= 1
