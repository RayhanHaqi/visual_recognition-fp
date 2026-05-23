"""Sliding-window tiling and multi-shift TTA offsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class TileWindow:
    x0: int
    y0: int
    x1: int
    y1: int
    shift_idx: int = 0


def iter_tile_windows(
    width: int,
    height: int,
    tile_size: int,
    stride: int | None = None,
    shifts: int = 1,
) -> Iterator[TileWindow]:
    """
    Yield tile windows covering the image.
    shifts: number of staggered grids (1 = single grid, up to 5 for TTA).
    """
    stride = stride or tile_size
    for shift_idx in range(max(1, shifts)):
        ox = (shift_idx * stride // max(1, shifts)) % max(1, stride)
        oy = (shift_idx * stride // (2 * max(1, shifts))) % max(1, stride)
        y = 0
        while y < height:
            x = 0
            while x < width:
                x0 = min(x + ox, max(0, width - tile_size))
                y0 = min(y + oy, max(0, height - tile_size))
                x1 = min(x0 + tile_size, width)
                y1 = min(y0 + tile_size, height)
                if x1 - x0 < tile_size // 2 or y1 - y0 < tile_size // 2:
                    x += stride
                    continue
                yield TileWindow(x0, y0, x1, y1, shift_idx)
                x += stride
            y += stride
        # trailing edge tiles
        if width >= tile_size:
            yield TileWindow(width - tile_size, max(0, min(oy, height - tile_size)),
                             width, min(tile_size + max(0, min(oy, height - tile_size)), height),
                             shift_idx)
        if height >= tile_size:
            yield TileWindow(max(0, min(ox, width - tile_size)), height - tile_size,
                             min(tile_size + max(0, min(ox, width - tile_size)), width), height,
                             shift_idx)


def tile_area_fraction(window: TileWindow, width: int, height: int) -> float:
    tw = window.x1 - window.x0
    th = window.y1 - window.y0
    return (tw * th) / max(1, width * height)


def aggregate_tile_predictions(
    preds: list[np.ndarray],
    windows: list[TileWindow],
    image_width: int,
    image_height: int,
    mode: str = "area_weighted",
) -> np.ndarray:
    """
    Combine per-tile 5-d count predictions into one image-level vector.
    """
    if not preds:
        return np.zeros(5, dtype=np.float32)

    stacked = np.stack([np.asarray(p, dtype=np.float64).reshape(5) for p in preds], axis=0)
    if mode == "sum":
        return np.maximum(stacked.sum(axis=0), 0).astype(np.float32)

    weights = np.array(
        [tile_area_fraction(w, image_width, image_height) for w in windows],
        dtype=np.float64,
    )
    if weights.sum() <= 0:
        weights = np.ones(len(weights)) / len(weights)
    else:
        weights /= weights.sum()
    out = (stacked * weights[:, None]).sum(axis=0)
    return np.maximum(out, 0).astype(np.float32)
