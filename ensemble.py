#!/usr/bin/env python3
"""Average multiple submission CSVs (same ids and columns)."""

import argparse
from pathlib import Path

import pandas as pd

from data.submission_ops import blend_submissions, write_submission

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", help="Submission CSV paths to average")
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    paths = [Path(c) if Path(c).is_absolute() else ROOT / c for c in args.csvs]
    frames = [pd.read_csv(path) for path in paths]
    out = blend_submissions(frames)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    write_submission(out, out_path)
    print(f"Ensembled {len(frames)} files -> {out_path}")


if __name__ == "__main__":
    main()
