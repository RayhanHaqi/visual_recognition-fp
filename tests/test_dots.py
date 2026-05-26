import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dots import (
    CLASS_TO_IDX,
    _classify_colors,
    counts_in_crop,
    extract_dots_from_pair,
    load_dot_cache,
)


def _write_pair(tmp_path: Path, name: str, dot_specs: list[tuple[int, int, tuple[int, int, int]]]):
    train = np.full((100, 100, 3), 180, dtype=np.uint8)
    dotted = train.copy()
    for x, y, rgb in dot_specs:
        cv2.circle(dotted, (x, y), 2, rgb, -1)
    train_path = tmp_path / f"{name}_train.jpg"
    dotted_path = tmp_path / f"{name}_dotted.jpg"
    cv2.imwrite(str(train_path), cv2.cvtColor(train, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(dotted_path), cv2.cvtColor(dotted, cv2.COLOR_RGB2BGR))
    return train, dotted


def test_extract_dots_from_synthetic_pair(tmp_path):
    train, dotted = _write_pair(
        tmp_path,
        "x",
        [(20, 20, (255, 0, 0)), (60, 60, (0, 255, 0))],
    )
    dots = extract_dots_from_pair(train, dotted)
    assert len(dots) == 2
    classes = sorted(d.class_idx for d in dots)
    assert classes == sorted([CLASS_TO_IDX["adult_males"], CLASS_TO_IDX["pups"]])


def test_extract_dots_finds_blue_and_red_small_dots():
    train = np.full((100, 100, 3), 120, dtype=np.uint8)
    dotted = train.copy()
    cv2.circle(dotted, (20, 20), 2, (255, 0, 0), -1)
    cv2.circle(dotted, (60, 60), 2, (0, 0, 255), -1)
    dots = extract_dots_from_pair(train, dotted)
    classes = sorted(d.class_idx for d in dots)
    assert classes == sorted([CLASS_TO_IDX["adult_males"], CLASS_TO_IDX["juveniles"]])


def test_extract_dots_keeps_faint_red_adult_male_dot():
    train = np.full((100, 100, 3), 120, dtype=np.uint8)
    dotted = train.copy()
    cv2.circle(dotted, (50, 50), 2, (255, 90, 90), -1)
    dots = extract_dots_from_pair(train, dotted)
    assert len(dots) == 1
    assert dots[0].class_idx == CLASS_TO_IDX["adult_males"]


def test_extract_dots_finds_faint_blue_and_brown_dots():
    train = np.full((100, 100, 3), 120, dtype=np.uint8)
    dotted = train.copy()
    cv2.circle(dotted, (25, 25), 2, (100, 100, 255), -1)
    cv2.circle(dotted, (75, 75), 2, (145, 45, 45), -1)
    dots = extract_dots_from_pair(train, dotted)
    classes = sorted(d.class_idx for d in dots)
    assert classes == sorted([CLASS_TO_IDX["juveniles"], CLASS_TO_IDX["adult_females"]])


def test_extract_dots_rejects_unsaturated_diff_blobs():
    train = np.full((100, 100, 3), 120, dtype=np.uint8)
    dotted = train.copy()
    dotted[20:45, 20:45] = np.array([140, 110, 90], dtype=np.uint8)
    dots = extract_dots_from_pair(train, dotted)
    assert dots == []


def test_extract_dots_rejects_unsaturated_reddish_terrain_patch():
    train = np.full((100, 100, 3), 120, dtype=np.uint8)
    dotted = train.copy()
    dotted[20:45, 20:45] = np.array([145, 105, 100], dtype=np.uint8)
    dots = extract_dots_from_pair(train, dotted)
    assert dots == []


def test_classify_colors_hsv_fallback_keeps_faint_red_not_terrain():
    assert _classify_colors(np.array([[255, 90, 90]], dtype=np.uint8)) == CLASS_TO_IDX[
        "adult_males"
    ]
    assert _classify_colors(np.array([[145, 105, 100]], dtype=np.uint8)) is None


def test_classify_colors_prefers_brown_female_over_red_hue():
    assert _classify_colors(np.array([[165, 42, 42]], dtype=np.uint8)) == CLASS_TO_IDX[
        "adult_females"
    ]


def test_extract_dots_keeps_saturated_brown_dot():
    train = np.full((100, 100, 3), 120, dtype=np.uint8)
    dotted = train.copy()
    cv2.circle(dotted, (50, 50), 2, (165, 42, 42), -1)
    dots = extract_dots_from_pair(train, dotted)
    assert len(dots) == 1
    assert dots[0].class_idx == CLASS_TO_IDX["adult_females"]


def test_extract_dots_rejects_large_diff_blob():
    train = np.full((100, 100, 3), 120, dtype=np.uint8)
    dotted = train.copy()
    cv2.rectangle(dotted, (10, 10), (70, 70), (255, 0, 0), -1)
    dots = extract_dots_from_pair(train, dotted)
    assert dots == []


def test_counts_in_crop():
    from data.dots import Dot

    dots = [
        Dot(10, 10, CLASS_TO_IDX["adult_males"]),
        Dot(15, 12, CLASS_TO_IDX["pups"]),
        Dot(80, 80, CLASS_TO_IDX["juveniles"]),
    ]
    counts = counts_in_crop(dots, 0, 0, 50)
    assert counts[CLASS_TO_IDX["adult_males"]] == 1.0
    assert counts[CLASS_TO_IDX["pups"]] == 1.0
    assert counts[CLASS_TO_IDX["juveniles"]] == 0.0


def test_load_dot_cache_roundtrip(tmp_path):
    cache = tmp_path / "dots.csv"
    cache.write_text(
        "image_id,x,y,class\n"
        "img_a,10,20,adult_males\n"
        "img_a,30,40,pups\n"
    )
    by_image = load_dot_cache(cache)
    assert len(by_image["img_a"]) == 2
    assert by_image["img_a"][0].class_idx == CLASS_TO_IDX["adult_males"]
