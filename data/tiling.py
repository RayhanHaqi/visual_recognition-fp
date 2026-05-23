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

    def key(self) -> tuple[int, int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1, self.shift_idx)


def iter_tile_windows(
    width: int,
    height: int,
    tile_size: int,
    stride: int | None = None,
    shifts: int = 1,
) -> Iterator[TileWindow]:
    """
    Yield unique tile windows covering the image (no duplicate coordinates per shift).
    shifts: number of staggered grids (1 = single grid, up to 5 for TTA).
    """
    stride = max(1, stride or tile_size)
    ts = tile_size

    for shift_idx in range(max(1, shifts)):
        ox = (shift_idx * stride // max(1, shifts)) % stride
        oy = (shift_idx * stride // (2 * max(1, shifts))) % stride
        seen: set[tuple[int, int, int, int, int]] = set()
        candidates: list[TileWindow] = []

        y = 0
        while y < height:
            x = 0
            while x < width:
                x0 = min(x + ox, max(0, width - ts))
                y0 = min(y + oy, max(0, height - ts))
                x1 = min(x0 + ts, width)
                y1 = min(y0 + ts, height)
                if x1 - x0 >= ts // 2 and y1 - y0 >= ts // 2:
                    candidates.append(TileWindow(x0, y0, x1, y1, shift_idx))
                x += stride
            y += stride

        if width >= ts:
            candidates.append(
                TileWindow(
                    width - ts,
                    max(0, min(oy, height - ts)),
                    width,
                    min(ts + max(0, min(oy, height - ts)), height),
                    shift_idx,
                )
            )
        if height >= ts:
            candidates.append(
                TileWindow(
                    max(0, min(ox, width - ts)),
                    height - ts,
                    min(ts + max(0, min(ox, width - ts)), width),
                    height,
                    shift_idx,
                )
            )

        for win in candidates:
            k = win.key()
            if k not in seen:
                seen.add(k)
                yield win


def tile_area_fraction(window: TileWindow, width: int, height: int) -> float:
    tw = window.x1 - window.x0
    th = window.y1 - window.y0
    return (tw * th) / max(1, width * height)


def aggregate_tile_predictions(
    preds: list[np.ndarray],
    windows: list[TileWindow],
    image_width: int,
    image_height: int,
    mode: str = "sum",
) -> np.ndarray:
    """
    Combine per-tile 5-d count predictions into one image-level vector.
    Default mode "sum" matches training contract (tile targets scale with area;
    summing tile predictions estimates image totals).
    """
    if not preds:
        return np.zeros(5, dtype=np.float32)
    if len(preds) != len(windows):
        raise ValueError(f"preds ({len(preds)}) and windows ({len(windows)}) length mismatch")

    stacked = np.stack([np.asarray(p, dtype=np.float64).reshape(5) for p in preds], axis=0)
    if mode == "sum":
        return np.maximum(stacked.sum(axis=0), 0).astype(np.float32)

    if mode == "area_weighted":
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

    raise ValueError(f"Unknown aggregation mode: {mode}")
