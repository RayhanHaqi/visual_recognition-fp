"""Extract per-dot labels from TrainDotted vs Train image pairs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.targets import COUNT_COLUMNS, list_train_images

ROOT = Path(__file__).resolve().parent.parent

# Competition dot colors (RGB), approximate.
CLASS_RGB = {
    "adult_males": np.array([255, 0, 0], dtype=np.float32),
    "subadult_males": np.array([255, 0, 255], dtype=np.float32),
    "adult_females": np.array([165, 42, 42], dtype=np.float32),
    "juveniles": np.array([0, 0, 255], dtype=np.float32),
    "pups": np.array([0, 255, 0], dtype=np.float32),
}

# Max RGB distance for saturated pixels to count as a class vote.
CLASS_MAX_DIST = {
    "adult_males": 90.0,
    "subadult_males": 90.0,
    "adult_females": 65.0,
    "juveniles": 90.0,
    "pups": 90.0,
}

CLASS_TO_IDX = {name: i for i, name in enumerate(COUNT_COLUMNS)}


@dataclass(frozen=True)
class Dot:
    x: int
    y: int
    class_idx: int


def _is_black_region(rgb: np.ndarray, threshold: int = 25) -> np.ndarray:
    return rgb.sum(axis=2) < threshold


def _nearest_class(pixel_rgb: np.ndarray) -> tuple[int, float]:
    best_idx = 0
    best_dist = float("inf")
    for name, ref in CLASS_RGB.items():
        dist = float(np.linalg.norm(pixel_rgb.astype(np.float32) - ref))
        if dist < best_dist:
            best_dist = dist
            best_idx = CLASS_TO_IDX[name]
    return best_idx, best_dist


def _classify_dot_color(rgb: np.ndarray) -> int:
    class_idx, _ = _nearest_class(rgb)
    return class_idx


def _classify_blob_pixels(
    dotted_rgb: np.ndarray,
    blob_mask: np.ndarray,
    min_saturation: int = 100,
    min_votes: int = 2,
    min_vote_fraction: float = 0.55,
) -> int | None:
    """
    Classify a diff blob using saturated dotted-image pixels inside the blob.
    Returns None when the blob does not look like a competition annotation dot.
    """
    ys, xs = np.where(blob_mask)
    if len(xs) == 0:
        return None

    colors = dotted_rgb[ys, xs].astype(np.uint8)
    hsv = cv2.cvtColor(colors.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sat_mask = hsv[:, 1] >= min_saturation
    if int(sat_mask.sum()) < min_votes:
        return None

    votes = np.zeros(len(COUNT_COLUMNS), dtype=np.int32)
    for pixel in colors[sat_mask]:
        class_idx, dist = _nearest_class(pixel)
        class_name = COUNT_COLUMNS[class_idx]
        if dist <= CLASS_MAX_DIST[class_name]:
            votes[class_idx] += 1

    total_votes = int(votes.sum())
    if total_votes < min_votes:
        return None

    winner = int(votes.argmax())
    if votes[winner] / total_votes < min_vote_fraction:
        return None
    return winner


def extract_dots_from_pair(
    train_rgb: np.ndarray,
    dotted_rgb: np.ndarray,
    min_diff: int = 45,
    min_blob_area: int = 2,
    max_blob_area: int = 80,
    min_saturation: int = 100,
    min_votes: int = 2,
) -> list[Dot]:
    """Find colored annotation dots via TrainDotted - Train difference."""
    if train_rgb.shape != dotted_rgb.shape:
        dotted_rgb = cv2.resize(
            dotted_rgb,
            (train_rgb.shape[1], train_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    diff = np.abs(dotted_rgb.astype(np.int16) - train_rgb.astype(np.int16)).sum(axis=2)
    black = _is_black_region(train_rgb) | _is_black_region(dotted_rgb)
    mask = (diff >= min_diff) & (~black)
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    dots: list[Dot] = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_blob_area or area > max_blob_area:
            continue
        blob_mask = labels == label
        class_idx = _classify_blob_pixels(
            dotted_rgb,
            blob_mask,
            min_saturation=min_saturation,
            min_votes=min_votes,
        )
        if class_idx is None:
            continue
        cx, cy = centroids[label]
        x, y = int(round(cx)), int(round(cy))
        if y < 0 or x < 0 or y >= dotted_rgb.shape[0] or x >= dotted_rgb.shape[1]:
            continue
        dots.append(Dot(x=x, y=y, class_idx=class_idx))
    return dots


def extract_dots_for_image(data_dir: Path, image_path: Path) -> list[Dot]:
    train_bgr = cv2.imread(str(image_path))
    if train_bgr is None:
        return []
    train_rgb = cv2.cvtColor(train_bgr, cv2.COLOR_BGR2RGB)
    dotted_path = data_dir / "TrainDotted" / image_path.name
    if not dotted_path.is_file():
        return []
    dotted_bgr = cv2.imread(str(dotted_path))
    if dotted_bgr is None:
        return []
    dotted_rgb = cv2.cvtColor(dotted_bgr, cv2.COLOR_BGR2RGB)
    return extract_dots_from_pair(train_rgb, dotted_rgb)


def counts_in_crop(dots: list[Dot], x0: int, y0: int, tile_size: int) -> np.ndarray:
    counts = np.zeros(len(COUNT_COLUMNS), dtype=np.float32)
    x1, y1 = x0 + tile_size, y0 + tile_size
    for dot in dots:
        if x0 <= dot.x < x1 and y0 <= dot.y < y1:
            counts[dot.class_idx] += 1.0
    return counts


def dots_to_rows(image_id: str, dots: list[Dot]) -> list[dict]:
    return [
        {"image_id": image_id, "x": d.x, "y": d.y, "class": COUNT_COLUMNS[d.class_idx]}
        for d in dots
    ]


def build_dot_cache(data_dir: Path, out_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    skipped = 0
    paths = list_train_images(data_dir)
    for path in tqdm(paths, desc="extract dots"):
        try:
            dots = extract_dots_for_image(data_dir, path)
        except Exception as exc:
            skipped += 1
            tqdm.write(f"skip {path.name}: {exc}")
            continue
        if not dots and not (data_dir / "TrainDotted" / path.name).is_file():
            skipped += 1
        rows.extend(dots_to_rows(path.stem, dots))
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if skipped:
        print(f"Skipped {skipped} images during dot extraction")
    return df


def load_dot_cache(cache_path: Path) -> dict[str, list[Dot]]:
    if not cache_path.is_file():
        return {}
    df = pd.read_csv(cache_path)
    by_image: dict[str, list[Dot]] = {}
    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        class_name = str(row["class"])
        if class_name not in CLASS_TO_IDX:
            continue
        dot = Dot(int(row["x"]), int(row["y"]), CLASS_TO_IDX[class_name])
        by_image.setdefault(image_id, []).append(dot)
    return by_image


def main() -> None:
    p = argparse.ArgumentParser(description="Build dot label cache from TrainDotted.")
    p.add_argument("--data_path", type=str, default="./datasets")
    p.add_argument("--output", type=str, default="./datasets/dot_labels.csv")
    args = p.parse_args()

    data_dir = Path(args.data_path)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    df = build_dot_cache(data_dir, out_path)
    print(f"Wrote {out_path} ({len(df)} dots from {df['image_id'].nunique()} images)")


if __name__ == "__main__":
    main()
