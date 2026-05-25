#!/usr/bin/env python3
"""Fetch competition train labels (not available via Kaggle API in 2026)."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "datasets" / "train.csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/LivingProgram/kaggle-sea-lion-data/"
    "master/correct_train.csv"
)

# Internal train.csv: id + official Kaggle count columns.
OUTPUT_COLUMNS = [
    "id",
    "adult_males",
    "subadult_males",
    "adult_females",
    "juveniles",
    "pups",
]


def fetch_train_csv(out_path: Path) -> None:
    import pandas as pd

    from data.targets import _normalize_train_columns

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading labels from {SOURCE_URL}", flush=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as resp:
        raw = resp.read()

    tmp = out_path.with_suffix(".csv.tmp")
    tmp.write_bytes(raw)
    try:
        df = _normalize_train_columns(pd.read_csv(tmp))
        missing = [c for c in OUTPUT_COLUMNS if c not in df.columns and c != "id"]
        # Allow train_id-only source files; juveniles must exist after normalize.
        if "juveniles" not in df.columns:
            raise ValueError("Downloaded labels missing juveniles column")
        for col in OUTPUT_COLUMNS[1:]:
            if col not in df.columns:
                raise ValueError(f"Downloaded labels missing column: {col}")
        df = df[OUTPUT_COLUMNS]
        df.to_csv(out_path, index=False)
    finally:
        tmp.unlink(missing_ok=True)

    print(f"Wrote {out_path} ({len(df)} rows)", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download corrected NOAA sea lion train labels to datasets/train.csv"
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        help="Output path (default: datasets/train.csv)",
    )
    args = p.parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    try:
        fetch_train_csv(out_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
