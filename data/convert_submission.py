"""Convert legacy submission CSV to official Kaggle format."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.targets import (
    COUNT_COLUMNS,
    SUBMISSION_COLUMNS,
    SUBMISSION_ID_COL,
    finalize_submission_df,
    normalize_test_id,
    submission_id_column,
)

ROOT = Path(__file__).resolve().parent.parent


def convert_submission(df: pd.DataFrame) -> pd.DataFrame:
    id_col = submission_id_column(df)

    if SUBMISSION_ID_COL in df.columns and all(c in df.columns for c in COUNT_COLUMNS):
        return finalize_submission_df(df)

    required_legacy = ["adult_males", "adult_females", "subadult_males", "subadult_females", "pups"]
    missing = [c for c in required_legacy if c not in df.columns]
    if missing:
        raise ValueError(
            f"Unrecognized submission format; missing columns: {missing}. "
            f"Expected Kaggle columns: {SUBMISSION_COLUMNS}"
        )

    out = pd.DataFrame(
        {
            SUBMISSION_ID_COL: df[id_col].map(normalize_test_id),
            "adult_males": df["adult_males"],
            "subadult_males": df["subadult_males"],
            "adult_females": df["adult_females"],
            "juveniles": df["subadult_females"],
            "pups": df["pups"],
        }
    )
    return finalize_submission_df(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert submission CSV to Kaggle format.")
    p.add_argument("input_csv", type=str)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    in_path = Path(args.input_csv)
    if not in_path.is_absolute():
        in_path = ROOT / in_path

    out_path = Path(args.output) if args.output else in_path.with_name(
        in_path.stem + "_kaggle.csv"
    )
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    df = convert_submission(pd.read_csv(in_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
