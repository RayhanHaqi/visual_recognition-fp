"""Expand partial submission to match official Kaggle sample_submission.csv."""

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


def expand_submission(
    partial: pd.DataFrame,
    template: pd.DataFrame,
) -> pd.DataFrame:
    id_col = submission_id_column(template)
    if id_col != SUBMISSION_ID_COL:
        template = template.rename(columns={id_col: SUBMISSION_ID_COL})

    partial_id_col = submission_id_column(partial)
    partial = partial.copy()
    partial["_key"] = partial[partial_id_col].map(normalize_test_id)

    pred_cols = [c for c in COUNT_COLUMNS if c in partial.columns]
    if not pred_cols:
        raise ValueError(f"Partial submission missing count columns: {COUNT_COLUMNS}")

    pred_by_id = partial.set_index("_key")[pred_cols]
    rows = []
    for test_id in template[SUBMISSION_ID_COL].map(normalize_test_id):
        if test_id in pred_by_id.index:
            counts = pred_by_id.loc[test_id]
            if isinstance(counts, pd.DataFrame):
                counts = counts.iloc[0]
            row = {SUBMISSION_ID_COL: test_id}
            for col in COUNT_COLUMNS:
                row[col] = counts.get(col, 0)
        else:
            row = {SUBMISSION_ID_COL: test_id, **{col: 0 for col in COUNT_COLUMNS}}
        rows.append(row)

    return finalize_submission_df(pd.DataFrame(rows))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Expand partial predictions to full official sample_submission row count."
    )
    p.add_argument("partial_csv", type=str, help="Submission with predictions for scored test ids")
    p.add_argument(
        "--template",
        type=str,
        default="./datasets/sample_submission.csv",
        help="Official Kaggle sample_submission.csv (defines required row count/order)",
    )
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    partial_path = Path(args.partial_csv)
    template_path = Path(args.template)
    out_path = Path(args.output)
    for path in (partial_path, template_path):
        if not path.is_absolute():
            path = ROOT / path
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    partial = pd.read_csv(partial_path if partial_path.is_absolute() else ROOT / partial_path)
    template = pd.read_csv(template_path if template_path.is_absolute() else ROOT / template_path)
    out = expand_submission(partial, template)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows, template={len(template)})")


if __name__ == "__main__":
    main()
