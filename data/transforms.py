"""Training augmentations."""

import random

import numpy as np
import torch
from torchvision import transforms as T


def build_train_transform(tile_size: int):
    return T.Compose([
        T.Resize((tile_size, tile_size)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(tile_size: int):
    return T.Compose([
        T.Resize((tile_size, tile_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def random_geom_augment(img: np.ndarray) -> np.ndarray:
    """Flip / 90-rot augment on HWC uint8/float image before PIL."""
    if random.random() < 0.5:
        img = np.fliplr(img).copy()
    if random.random() < 0.5:
        img = np.flipud(img).copy()
    k = random.randint(0, 3)
    if k:
        img = np.rot90(img, k).copy()
    return img


def random_scale(img: np.ndarray, low: float = 0.9, high: float = 1.1) -> np.ndarray:
    import cv2

    scale = random.uniform(low, high)
    h, w = img.shape[:2]
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
