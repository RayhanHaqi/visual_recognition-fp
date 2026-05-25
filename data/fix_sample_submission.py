"""Rebuild datasets/sample_submission.csv from Test/*.jpg (official Kaggle layout)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.targets import COUNT_COLUMNS, SUBMISSION_COLUMNS, SUBMISSION_ID_COL, list_test_images, normalize_test_id

ROOT = Path(__file__).resolve().parent.parent


def build_sample(data_dir: Path, test_subdir: str = "Test", limit: int = 100) -> pd.DataFrame:
    paths = list_test_images(data_dir, test_subdir=test_subdir)
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No test images under {data_dir / test_subdir}")

    df = pd.DataFrame({SUBMISSION_ID_COL: [normalize_test_id(p.stem) for p in paths]})
    for col in COUNT_COLUMNS:
        df[col] = 0.0
    return df[SUBMISSION_COLUMNS]


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild sample_submission.csv from Test images.")
    p.add_argument("--data_path", type=str, default="./datasets")
    p.add_argument("--test_subdir", type=str, default="Test")
    p.add_argument("--limit", type=int, default=100, help="Max test rows (0 = all)")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    data_dir = Path(args.data_path)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    df = build_sample(data_dir, test_subdir=args.test_subdir, limit=args.limit)
    out = Path(args.output) if args.output else data_dir / "sample_submission.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
