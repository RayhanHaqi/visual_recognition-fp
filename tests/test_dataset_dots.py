import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import SeaLionTileDataset
from data.dots import Dot, CLASS_TO_IDX
from data.targets import load_train_counts


def test_tile_dataset_dots_mode_uses_crop_counts(mini_data_dir):
    counts_df = load_train_counts(mini_data_dir, exclude_mismatched=False)
    paths = sorted((mini_data_dir / "Train").glob("*.jpg"))[:1]
    dots_by_image = {
        paths[0].stem: [
            Dot(50, 50, CLASS_TO_IDX["adult_males"]),
            Dot(55, 52, CLASS_TO_IDX["pups"]),
        ]
    }
    ds = SeaLionTileDataset(
        paths,
        counts_df,
        tile_size=128,
        tiles_per_image=1,
        train=False,
        label_mode="dots",
        dots_by_image=dots_by_image,
    )
    tensor, target, img_id = ds[0]
    assert tensor.shape[0] == 3
    assert target.shape == (5,)
    assert float(target[CLASS_TO_IDX["adult_males"]]) >= 0.0
