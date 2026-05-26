"""PyTorch datasets for image-level and tile-level training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.dots import Dot, counts_in_crop
from data.targets import (
    COUNT_COLUMNS,
    counts_for_image,
    list_train_images,
    load_train_counts,
)
from data.transforms import build_eval_transform, build_train_transform, random_geom_augment


class SeaLionImageDataset(Dataset):
    """Resize whole image to tile_size; target = image-level counts."""

    def __init__(
        self,
        image_paths: list[Path],
        counts_df,
        tile_size: int = 299,
        train: bool = True,
        augment_geom: bool = True,
    ):
        self.paths = image_paths
        self.counts_df = counts_df
        self.train = train
        self.augment_geom = augment_geom and train
        self.transform = build_train_transform(tile_size) if train else build_eval_transform(tile_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img_id = path.stem
        img = np.array(Image.open(path).convert("RGB"))
        if self.augment_geom:
            img = random_geom_augment(img)
        tensor = self.transform(Image.fromarray(img))
        target = torch.from_numpy(counts_for_image(self.counts_df, img_id))
        return tensor, target, img_id


class SeaLionTileDataset(Dataset):
    """
    Random crops from training images.
    label_mode='area': target = image counts * crop area fraction.
    label_mode='dots': target = dot counts inside crop (from TrainDotted cache).
    """

    def __init__(
        self,
        image_paths: list[Path],
        counts_df,
        tile_size: int = 299,
        tiles_per_image: int = 8,
        train: bool = True,
        label_mode: str = "area",
        dots_by_image: dict | None = None,
        balanced_positive_fraction: float = 0.5,
        background_tries: int = 20,
    ):
        self.paths = image_paths
        self.counts_df = counts_df
        self.tile_size = tile_size
        self.tiles_per_image = tiles_per_image
        self.train = train
        self.label_mode = label_mode
        self.dots_by_image = dots_by_image or {}
        self.balanced_positive_fraction = balanced_positive_fraction
        self.background_tries = background_tries
        self.transform = build_train_transform(tile_size) if train else build_eval_transform(tile_size)
        self._length = len(image_paths) * tiles_per_image

    def __len__(self):
        return self._length

    def _random_crop_origin(self, width: int, height: int) -> tuple[int, int]:
        ts = self.tile_size
        if width < ts or height < ts:
            return 0, 0
        x0 = np.random.randint(0, width - ts + 1) if self.train else (width - ts) // 2
        y0 = np.random.randint(0, height - ts + 1) if self.train else (height - ts) // 2
        return int(x0), int(y0)

    def _dot_centered_crop_origin(self, dot: Dot, width: int, height: int) -> tuple[int, int]:
        ts = self.tile_size
        if width < ts or height < ts:
            return 0, 0
        jitter_x = np.random.randint(-ts // 4, ts // 4 + 1) if self.train else 0
        jitter_y = np.random.randint(-ts // 4, ts // 4 + 1) if self.train else 0
        x0 = int(np.clip(dot.x - ts // 2 + jitter_x, 0, width - ts))
        y0 = int(np.clip(dot.y - ts // 2 + jitter_y, 0, height - ts))
        return x0, y0

    def _background_crop_origin(
        self,
        dots: list[Dot],
        width: int,
        height: int,
    ) -> tuple[int, int]:
        ts = self.tile_size
        if width < ts or height < ts:
            return 0, 0
        for _ in range(self.background_tries):
            x0, y0 = self._random_crop_origin(width, height)
            if counts_in_crop(dots, x0, y0, ts).sum() == 0:
                return x0, y0
        return self._random_crop_origin(width, height)

    def _balanced_crop_origin(
        self,
        img_id: str,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        dots = self.dots_by_image.get(img_id, [])
        want_positive = bool(dots) and np.random.random() < self.balanced_positive_fraction
        if want_positive:
            dot = dots[np.random.randint(0, len(dots))]
            return self._dot_centered_crop_origin(dot, width, height)
        return self._background_crop_origin(dots, width, height)

    def __getitem__(self, idx: int):
        path = self.paths[idx % len(self.paths)]
        img_id = path.stem
        img = np.array(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]
        ts = self.tile_size

        if w >= ts and h >= ts:
            if self.label_mode == "balanced_dots":
                x0, y0 = self._balanced_crop_origin(img_id, w, h)
            else:
                x0, y0 = self._random_crop_origin(w, h)
            crop = img[y0 : y0 + ts, x0 : x0 + ts]
            frac = (ts * ts) / max(1, w * h)
        else:
            x0, y0 = 0, 0
            crop = img
            frac = 1.0

        if self.train:
            crop = random_geom_augment(crop)

        tensor = self.transform(Image.fromarray(crop))
        if self.label_mode in {"dots", "balanced_dots"}:
            dots = self.dots_by_image.get(img_id, [])
            target_np = counts_in_crop(dots, x0, y0, ts)
        else:
            full = counts_for_image(self.counts_df, img_id)
            target_np = (full * frac).astype(np.float32)
        target = torch.from_numpy(target_np)
        return tensor, target, img_id


def build_train_val_paths(data_dir: Path, val_frac: float, seed: int = 42):
    from utils.splits import train_val_split

    paths = list_train_images(data_dir)
    train_paths, val_paths = train_val_split(paths, val_frac=val_frac, seed=seed)
    counts_df = load_train_counts(data_dir)
    return train_paths, val_paths, counts_df
