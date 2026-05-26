#!/usr/bin/env python3
"""Check how many sample_submission test_ids map to files under Test/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.targets import normalize_test_id, submission_id_column


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose sample_submission vs Test image ids.")
    p.add_argument("--data_path", type=str, default="./datasets")
    p.add_argument("--test_subdir", type=str, default="Test_scaled_0.5")
    p.add_argument("--sample", type=str, default=None, help="Default: datasets/sample_submission.csv")
    args = p.parse_args()

    data_dir = Path(args.data_path)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    sample_path = Path(args.sample) if args.sample else data_dir / "sample_submission.csv"
    if not sample_path.is_absolute():
        sample_path = ROOT / sample_path

    test_dir = data_dir / args.test_subdir
    if not sample_path.is_file():
        raise SystemExit(f"Missing {sample_path}")
    if not test_dir.is_dir():
        raise SystemExit(f"Missing {test_dir}")

    sample = pd.read_csv(sample_path)
    id_col = submission_id_column(sample)
    stems = {p.stem for p in test_dir.glob("*.jpg")}

    def has_file(tid) -> bool:
        return str(normalize_test_id(tid)) in stems

    matched = sample[id_col].map(has_file)
    n_match = int(matched.sum())
    n_rows = len(sample)
    n_unique = int(sample[id_col].nunique())

    print(f"sample: {sample_path}")
    print(f"test dir: {test_dir} ({len(stems)} jpg)")
    print(f"sample rows: {n_rows}")
    print(f"unique {id_col}: {n_unique}")
    print(f"rows with a {args.test_subdir} file: {n_match}")
    print(f"rows without file: {n_rows - n_match}")
    if n_match > 500:
        print(
            "\nLikely rebuilt sample (1 row per jpg) — inference will be very slow.\n"
            "Restore official sample_submission.csv from the competition 7z archive."
        )
    elif n_match < 200:
        print(
            "\nLooks like official layout — inference should use ~100 model runs, "
            "zero-fill the rest, still write 18636 submission rows."
        )


if __name__ == "__main__":
    main()
