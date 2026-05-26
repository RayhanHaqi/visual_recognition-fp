#!/usr/bin/env python3
"""Remove mismatched train images and optionally downscale test images."""

import argparse
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from data.submission_ops import scaled_test_subdir

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "datasets"


def load_mismatched(data_dir: Path) -> set[str]:
    path = data_dir / "MismatchedTrainImages.txt"
    if not path.is_file():
        return set()
    names = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            names.add(line)
    return names


def remove_mismatched(data_dir: Path, dry_run: bool = False) -> int:
    bad = load_mismatched(data_dir)
    if not bad:
        print("No mismatched list or empty.")
        return 0
    removed = 0
    for sub in ("Train", "TrainDotted"):
        folder = data_dir / sub
        if not folder.is_dir():
            continue
        for name in bad:
            for ext in ("", ".jpg", ".JPG"):
                p = folder / (name if name.endswith(".jpg") else f"{name}{ext}")
                if not p.suffix:
                    p = folder / f"{name}.jpg"
                if p.is_file():
                    if dry_run:
                        print(f"would remove: {p}")
                    else:
                        p.unlink()
                    removed += 1
    print(f"Removed {removed} mismatched file(s) from Train/TrainDotted.")
    return removed


def downscale_test(data_dir: Path, scale: float, out_subdir: str = "Test_scaled"):
    src = data_dir / "Test"
    dst = data_dir / out_subdir
    if not src.is_dir():
        raise FileNotFoundError(f"Missing {src}")
    dst.mkdir(parents=True, exist_ok=True)
    images = sorted(src.glob("*.jpg"))
    for path in tqdm(images, desc=f"Downscale Test x{scale}"):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst / path.name), small)
    print(f"Wrote {len(images)} images to {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default=str(DEFAULT_DATA))
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--downscale_test", type=float, default=None,
                   help="e.g. 0.5 to halve each Test dimension")
    p.add_argument(
        "--out_subdir",
        type=str,
        default=None,
        help="Output subdir under datasets/ (default: Test_scaled_<scale> when downscaling)",
    )
    args = p.parse_args()
    data_dir = Path(args.data_path)

    remove_mismatched(data_dir, dry_run=args.dry_run)
    if args.downscale_test:
        out_subdir = args.out_subdir or scaled_test_subdir(args.downscale_test)
        downscale_test(data_dir, args.downscale_test, out_subdir=out_subdir)


if __name__ == "__main__":
    main()
