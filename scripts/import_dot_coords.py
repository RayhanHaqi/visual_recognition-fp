#!/usr/bin/env python3
"""Convert community dot coordinate CSVs into FP dot_labels.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.targets import COUNT_COLUMNS

ROOT = Path(__file__).resolve().parent.parent

# lopuhin/kaggle-lions-2017 (2nd place): tid,cls,row,col with cls 0..4
LOPUHIN_CLS_TO_NAME = dict(enumerate(COUNT_COLUMNS))


def lopuhin_to_dot_labels(df: pd.DataFrame) -> pd.DataFrame:
    required = {"tid", "cls", "row", "col"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Expected columns {sorted(required)}, got {list(df.columns)}")

    out = pd.DataFrame(
        {
            "image_id": df["tid"].astype(int).astype(str),
            "x": df["col"].astype(int),
            "y": df["row"].astype(int),
            "class": df["cls"].astype(int).map(LOPUHIN_CLS_TO_NAME),
        }
    )
    unknown = out["class"].isna()
    if unknown.any():
        bad = sorted(df.loc[unknown, "cls"].unique().tolist())
        raise ValueError(f"Unknown lopuhin cls values: {bad}")
    return out


def import_lopuhin_coords(coords_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(coords_path)
    out = lopuhin_to_dot_labels(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Import lopuhin coords-threeplusone-v0.4.csv -> datasets/dot_labels.csv"
    )
    p.add_argument(
        "--coords",
        type=str,
        default="./data/coords-threeplusone-v0.4.csv",
        help="Source CSV (committed copy from lopuhin/kaggle-lions-2017)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="./datasets/dot_labels.csv",
        help="Destination dot cache for train.py / analyze_dot_cache.py",
    )
    args = p.parse_args()

    coords_path = Path(args.coords)
    if not coords_path.is_absolute():
        coords_path = ROOT / coords_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if not coords_path.is_file():
        raise SystemExit(
            f"Missing {coords_path}. Use the committed file under data/ or download from\n"
            "https://github.com/lopuhin/kaggle-lions-2017/blob/master/data/coords-threeplusone-v0.4.csv"
        )

    out = import_lopuhin_coords(coords_path, output_path)
    n_images = out["image_id"].nunique()
    print(f"Wrote {output_path} ({len(out)} dots from {n_images} images)")


if __name__ == "__main__":
    main()
