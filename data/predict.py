"""Shared tiled inference: per-tile counts summed to image-level prediction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False

from data.tiling import aggregate_tile_predictions, iter_tile_windows

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Canonical contract: model outputs per-tile counts; image prediction = sum of unique tiles.
DEFAULT_AGGREGATE_MODE = "sum"

_TO_TENSOR = T.ToTensor()
_NORMALIZE = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)


@dataclass
class InferenceTimings:
    load_image_sec: float = 0.0
    generate_windows_sec: float = 0.0
    preprocess_sec: float = 0.0
    h2d_sec: float = 0.0
    forward_sec: float = 0.0
    aggregate_sec: float = 0.0
    n_images: int = 0
    n_tiles: int = 0
    n_batch_forwards: int = 0
    total_image_sec: float = 0.0

    def add_image(self, n_tiles: int, elapsed_sec: float) -> None:
        self.n_images += 1
        self.n_tiles += n_tiles
        self.total_image_sec += elapsed_sec

    def merge(self, other: InferenceTimings) -> None:
        for name in (
            "load_image_sec",
            "generate_windows_sec",
            "preprocess_sec",
            "h2d_sec",
            "forward_sec",
            "aggregate_sec",
            "total_image_sec",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.n_images += other.n_images
        self.n_tiles += other.n_tiles
        self.n_batch_forwards += other.n_batch_forwards


def format_inference_profile(timings: InferenceTimings) -> dict[str, float]:
    total_tracked = (
        timings.load_image_sec
        + timings.generate_windows_sec
        + timings.preprocess_sec
        + timings.h2d_sec
        + timings.forward_sec
        + timings.aggregate_sec
    )
    denom = total_tracked if total_tracked > 0 else 1.0
    elapsed = timings.total_image_sec if timings.total_image_sec > 0 else total_tracked
    images_per_sec = timings.n_images / elapsed if elapsed > 0 and timings.n_images else 0.0
    tiles_per_sec = timings.n_tiles / elapsed if elapsed > 0 and timings.n_tiles else 0.0
    return {
        "load_image_sec": timings.load_image_sec,
        "generate_windows_sec": timings.generate_windows_sec,
        "preprocess_sec": timings.preprocess_sec,
        "h2d_sec": timings.h2d_sec,
        "forward_sec": timings.forward_sec,
        "aggregate_sec": timings.aggregate_sec,
        "total_tracked_sec": total_tracked,
        "total_image_sec": elapsed,
        "n_images": float(timings.n_images),
        "n_tiles": float(timings.n_tiles),
        "n_batch_forwards": float(timings.n_batch_forwards),
        "images_per_sec": images_per_sec,
        "tiles_per_sec": tiles_per_sec,
        "load_pct": 100.0 * timings.load_image_sec / denom,
        "windows_pct": 100.0 * timings.generate_windows_sec / denom,
        "preprocess_pct": 100.0 * timings.preprocess_sec / denom,
        "h2d_pct": 100.0 * timings.h2d_sec / denom,
        "forward_pct": 100.0 * timings.forward_sec / denom,
        "aggregate_pct": 100.0 * timings.aggregate_sec / denom,
    }


def _preprocess_crop(crop: np.ndarray, tile_size: int) -> torch.Tensor:
    if _HAS_CV2:
        resized = cv2.resize(crop, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        tensor = _NORMALIZE(_TO_TENSOR(resized))
    else:
        pil = Image.fromarray(crop).resize((tile_size, tile_size), Image.BILINEAR)
        tensor = _NORMALIZE(_TO_TENSOR(pil))
    return tensor


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward_batch(
    model: torch.nn.Module,
    tensors: list[torch.Tensor],
    device: torch.device,
    timings: InferenceTimings | None = None,
    use_amp: bool = False,
) -> list[np.ndarray]:
    if timings is not None:
        t0 = time.perf_counter()
    batch = torch.stack(tensors)
    if device.type == "cuda":
        batch = batch.pin_memory()
    batch = batch.to(device, non_blocking=device.type == "cuda")
    if timings is not None:
        timings.h2d_sec += time.perf_counter() - t0

    if timings is not None:
        t1 = time.perf_counter()
    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda"):
            out = torch.clamp(model(batch), min=0)
    else:
        out = torch.clamp(model(batch), min=0)
    _sync_if_cuda(device)
    if timings is not None:
        timings.forward_sec += time.perf_counter() - t1
        timings.n_batch_forwards += 1

    out_np = out.cpu().numpy()
    return [out_np[i] for i in range(out_np.shape[0])]


def _predict_from_windows(
    model: torch.nn.Module,
    img: np.ndarray,
    windows: list,
    tile_size: int,
    device: torch.device,
    batch_size: int,
    image_width: int,
    image_height: int,
    timings: InferenceTimings | None = None,
    use_amp: bool = False,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    batch_tensors: list[torch.Tensor] = []

    for win in windows:
        if timings is not None:
            t0 = time.perf_counter()
        crop = img[win.y0 : win.y1, win.x0 : win.x1]
        batch_tensors.append(_preprocess_crop(crop, tile_size))
        if timings is not None:
            timings.preprocess_sec += time.perf_counter() - t0

        if len(batch_tensors) >= batch_size:
            preds.extend(_forward_batch(model, batch_tensors, device, timings, use_amp))
            batch_tensors = []

    if batch_tensors:
        preds.extend(_forward_batch(model, batch_tensors, device, timings, use_amp))

    if timings is not None:
        t0 = time.perf_counter()
    result = aggregate_tile_predictions(
        preds, windows, image_width, image_height, mode=DEFAULT_AGGREGATE_MODE
    )
    if timings is not None:
        timings.aggregate_sec += time.perf_counter() - t0
    return result


@torch.inference_mode()
def predict_image_tiled(
    model: torch.nn.Module,
    image_path: Path | str,
    tile_size: int,
    device: torch.device,
    shifts: int = 5,
    stride: int | None = None,
    batch_size: int = 8,
    timings: InferenceTimings | None = None,
    use_amp: bool = False,
) -> np.ndarray:
    """
    Run tiled inference on one image. Returns 5-d non-negative counts (image level).
    Multiple shifts: sum tiles within each shift grid, then average shift totals (TTA).
    """
    image_t0 = time.perf_counter()
    model.eval()
    per_image = InferenceTimings() if timings is not None else None
    active = per_image if per_image is not None else timings

    if active is not None:
        t0 = time.perf_counter()
    img = np.array(Image.open(image_path).convert("RGB"))
    if active is not None:
        active.load_image_sec += time.perf_counter() - t0

    h, w = img.shape[:2]
    stride = stride if stride is not None else tile_size
    shift_count = max(1, shifts)

    if active is not None:
        t0 = time.perf_counter()
    windows = list(iter_tile_windows(w, h, tile_size, stride=stride, shifts=shift_count))
    if active is not None:
        active.generate_windows_sec += time.perf_counter() - t0

    if not windows:
        if active is not None:
            t0 = time.perf_counter()
        tensor = _preprocess_crop(img, tile_size)
        if device.type == "cuda":
            tensor = tensor.pin_memory()
        tensor = tensor.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
        if active is not None:
            active.preprocess_sec += time.perf_counter() - t0
            active.n_tiles = 1
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                result = torch.clamp(model(tensor), min=0).cpu().numpy()[0]
        else:
            result = torch.clamp(model(tensor), min=0).cpu().numpy()[0]
        if timings is not None and per_image is not None:
            per_image.n_batch_forwards = 1
            per_image.add_image(
                n_tiles=per_image.n_tiles,
                elapsed_sec=time.perf_counter() - image_t0,
            )
            timings.merge(per_image)
        return result

    if shift_count == 1:
        result = _predict_from_windows(
            model, img, windows, tile_size, device, batch_size, w, h, active, use_amp
        )
        if timings is not None and per_image is not None:
            per_image.n_tiles = len(windows)
            per_image.add_image(
                n_tiles=per_image.n_tiles,
                elapsed_sec=time.perf_counter() - image_t0,
            )
            timings.merge(per_image)
        return result

    by_shift: dict[int, list] = {}
    for win in windows:
        by_shift.setdefault(win.shift_idx, []).append(win)

    shift_totals = [
        _predict_from_windows(
            model, img, by_shift[shift_idx], tile_size, device, batch_size, w, h, active, use_amp
        )
        for shift_idx in sorted(by_shift)
    ]

    if active is not None:
        t0 = time.perf_counter()
    result = np.mean(np.stack(shift_totals, axis=0), axis=0).astype(np.float32)
    if active is not None:
        active.aggregate_sec += time.perf_counter() - t0
        active.n_tiles = len(windows)
        active.add_image(
            n_tiles=active.n_tiles,
            elapsed_sec=time.perf_counter() - image_t0,
        )
        timings.merge(per_image)

    return result
