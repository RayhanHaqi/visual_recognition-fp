"""PyTorch datasets for image-level and tile-level training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.dots import Dot, counts_in_crop, gaussian_counts_in_crop
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

    Optional scale augmentation (Asanakoy-style): sample a larger/smaller source crop,
    count dots in that window, then resize to tile_size for the model input.
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
        scale_min: float | None = None,
        scale_max: float | None = None,
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
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.transform = build_train_transform(tile_size) if train else build_eval_transform(tile_size)
        self._length = len(image_paths) * tiles_per_image

    def __len__(self):
        return self._length

    def _use_scale_augment(self) -> bool:
        return (
            self.train
            and self.scale_min is not None
            and self.scale_max is not None
            and self.scale_max > self.scale_min > 0
            and self.label_mode in {"dots", "balanced_dots", "gaussian_dots"}
        )

    def _sample_source_size(self, width: int, height: int) -> int:
        ts = self.tile_size
        if not self._use_scale_augment():
            return ts
        scale = float(np.random.uniform(self.scale_min, self.scale_max))
        sz = int(round(ts / scale))
        sz = max(8, min(sz, width, height))
        return max(ts, sz) if width >= ts and height >= ts else min(width, height)

    def _random_crop_origin(self, width: int, height: int, crop_size: int) -> tuple[int, int]:
        if width < crop_size or height < crop_size:
            return 0, 0
        x0 = np.random.randint(0, width - crop_size + 1) if self.train else (width - crop_size) // 2
        y0 = np.random.randint(0, height - crop_size + 1) if self.train else (height - crop_size) // 2
        return int(x0), int(y0)

    def _dot_centered_crop_origin(
        self, dot: Dot, width: int, height: int, crop_size: int
    ) -> tuple[int, int]:
        if width < crop_size or height < crop_size:
            return 0, 0
        jitter_span = max(1, crop_size // 4)
        jitter_x = np.random.randint(-jitter_span, jitter_span + 1) if self.train else 0
        jitter_y = np.random.randint(-jitter_span, jitter_span + 1) if self.train else 0
        x0 = int(np.clip(dot.x - crop_size // 2 + jitter_x, 0, width - crop_size))
        y0 = int(np.clip(dot.y - crop_size // 2 + jitter_y, 0, height - crop_size))
        return x0, y0

    def _background_crop_origin(
        self,
        dots: list[Dot],
        width: int,
        height: int,
        crop_size: int,
    ) -> tuple[int, int]:
        if width < crop_size or height < crop_size:
            return 0, 0
        for _ in range(self.background_tries):
            x0, y0 = self._random_crop_origin(width, height, crop_size)
            if counts_in_crop(dots, x0, y0, crop_size).sum() == 0:
                return x0, y0
        return self._random_crop_origin(width, height, crop_size)

    def _balanced_crop_origin(
        self,
        img_id: str,
        width: int,
        height: int,
        crop_size: int,
    ) -> tuple[int, int]:
        dots = self.dots_by_image.get(img_id, [])
        want_positive = bool(dots) and np.random.random() < self.balanced_positive_fraction
        if want_positive:
            dot = dots[np.random.randint(0, len(dots))]
            return self._dot_centered_crop_origin(dot, width, height, crop_size)
        return self._background_crop_origin(dots, width, height, crop_size)

    def __getitem__(self, idx: int):
        path = self.paths[idx % len(self.paths)]
        img_id = path.stem
        img = np.array(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]
        ts = self.tile_size
        crop_size = self._sample_source_size(w, h)

        if w >= crop_size and h >= crop_size:
            if self.label_mode in {"balanced_dots", "gaussian_dots"}:
                x0, y0 = self._balanced_crop_origin(img_id, w, h, crop_size)
            else:
                x0, y0 = self._random_crop_origin(w, h, crop_size)
            crop = img[y0 : y0 + crop_size, x0 : x0 + crop_size]
            frac = (crop_size * crop_size) / max(1, w * h)
        else:
            x0, y0 = 0, 0
            crop = img
            crop_size = min(h, w)
            frac = 1.0

        if self.train:
            crop = random_geom_augment(crop)

        tensor = self.transform(Image.fromarray(crop))
        if self.label_mode in {"dots", "balanced_dots", "gaussian_dots"}:
            dots = self.dots_by_image.get(img_id, [])
            if self.label_mode == "gaussian_dots":
                target_np = gaussian_counts_in_crop(dots, x0, y0, crop_size)
            else:
                target_np = counts_in_crop(dots, x0, y0, crop_size)
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
