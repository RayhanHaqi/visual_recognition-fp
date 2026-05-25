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
COUNT_COLUMNS = [
    "id",
    "adult_males",
    "adult_females",
    "subadult_males",
    "subadult_females",
    "pups",
]


def fetch_train_csv(out_path: Path) -> None:
    import pandas as pd

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading labels from {SOURCE_URL}", flush=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as resp:
        raw = resp.read()

    tmp = out_path.with_suffix(".csv.tmp")
    tmp.write_bytes(raw)
    try:
        df = pd.read_csv(tmp)
        df = df.rename(columns={"train_id": "id", "juveniles": "subadult_females"})
        missing = [c for c in COUNT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Downloaded labels missing columns: {missing}")
        df = df[COUNT_COLUMNS]
        df["id"] = df["id"].astype(str)
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
