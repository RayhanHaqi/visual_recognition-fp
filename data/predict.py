"""Shared tiled inference: per-tile counts summed to image-level prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from data.tiling import aggregate_tile_predictions, iter_tile_windows

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Canonical contract: model outputs per-tile counts; image prediction = sum of unique tiles.
DEFAULT_AGGREGATE_MODE = "sum"


def _normalize():
    return T.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def _preprocess_crop(crop: np.ndarray, tile_size: int) -> torch.Tensor:
    pil = Image.fromarray(crop).resize((tile_size, tile_size), Image.BILINEAR)
    return _normalize()(T.ToTensor()(pil))


def _forward_batch(
    model: torch.nn.Module, tensors: list[torch.Tensor], device: torch.device
) -> list[np.ndarray]:
    batch = torch.stack(tensors).to(device)
    out = torch.clamp(model(batch), min=0).cpu().numpy()
    return [out[i] for i in range(out.shape[0])]


def _predict_from_windows(
    model: torch.nn.Module,
    img: np.ndarray,
    windows: list,
    tile_size: int,
    device: torch.device,
    batch_size: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    batch_tensors: list[torch.Tensor] = []

    for win in windows:
        crop = img[win.y0 : win.y1, win.x0 : win.x1]
        batch_tensors.append(_preprocess_crop(crop, tile_size))
        if len(batch_tensors) >= batch_size:
            preds.extend(_forward_batch(model, batch_tensors, device))
            batch_tensors = []

    if batch_tensors:
        preds.extend(_forward_batch(model, batch_tensors, device))

    return aggregate_tile_predictions(
        preds, windows, image_width, image_height, mode=DEFAULT_AGGREGATE_MODE
    )


@torch.no_grad()
def predict_image_tiled(
    model: torch.nn.Module,
    image_path: Path | str,
    tile_size: int,
    device: torch.device,
    shifts: int = 5,
    stride: int | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Run tiled inference on one image. Returns 5-d non-negative counts (image level).
    Multiple shifts: sum tiles within each shift grid, then average shift totals (TTA).
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    stride = stride or tile_size // 2
    shift_count = max(1, shifts)
    windows = list(iter_tile_windows(w, h, tile_size, stride=stride, shifts=shift_count))

    if not windows:
        t = _preprocess_crop(img, tile_size).unsqueeze(0).to(device)
        model.eval()
        return torch.clamp(model(t), min=0).cpu().numpy()[0]

    if shift_count == 1:
        return _predict_from_windows(model, img, windows, tile_size, device, batch_size, w, h)

    by_shift: dict[int, list] = {}
    for win in windows:
        by_shift.setdefault(win.shift_idx, []).append(win)

    shift_totals = [
        _predict_from_windows(
            model, img, by_shift[shift_idx], tile_size, device, batch_size, w, h
        )
        for shift_idx in sorted(by_shift)
    ]
    return np.mean(np.stack(shift_totals, axis=0), axis=0).astype(np.float32)
