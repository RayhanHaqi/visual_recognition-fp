#!/usr/bin/env python3
"""Average multiple submission CSVs (same ids and columns)."""

import argparse
from pathlib import Path

import pandas as pd

from data.targets import COUNT_COLUMNS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+", help="Submission CSV paths to average")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--sample", type=str, default="./datasets/sample_submission.csv")
    args = p.parse_args()

    sample = pd.read_csv(args.sample)
    id_col = "id" if "id" in sample.columns else sample.columns[0]
    cols = COUNT_COLUMNS

    dfs = [pd.read_csv(c) for c in args.csvs]
    base = dfs[0].set_index(id_col)
    for df in dfs[1:]:
        base = base.add(df.set_index(id_col)[cols], fill_value=0)
    base[cols] = base[cols] / len(dfs)
    base[cols] = base[cols].clip(lower=0)
    out = base.reset_index()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Ensembled {len(dfs)} files -> {out_path}")


if __name__ == "__main__":
    main()
