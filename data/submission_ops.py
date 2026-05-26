"""Canonical helpers for loading, blending, scaling, and writing submission CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.targets import (
    COUNT_COLUMNS,
    SUBMISSION_ID_COL,
    finalize_submission_df,
    normalize_test_id,
    submission_id_column,
)


def scaled_test_subdir(scale: float) -> str:
    """Directory name for downscaled test images at a given scale factor."""
    return f"Test_scaled_{scale:g}"


def load_submission(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ids and count columns; reject duplicate test_id rows."""
    id_col = submission_id_column(df)
    missing = [col for col in COUNT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Submission missing count columns: {missing}")

    out = df[[id_col, *COUNT_COLUMNS]].copy()
    out = out.rename(columns={id_col: SUBMISSION_ID_COL})
    out[SUBMISSION_ID_COL] = out[SUBMISSION_ID_COL].map(normalize_test_id)
    for col in COUNT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="raise").astype(float)

    if out[SUBMISSION_ID_COL].duplicated().any():
        raise ValueError("Duplicate test_id rows in submission")
    return out


def apply_scales(df: pd.DataFrame, scales: dict[str, float] | None = None) -> pd.DataFrame:
    """Multiply selected count columns; input must already be load_submission()-clean."""
    out = df.copy()
    for col, scale in (scales or {}).items():
        if col not in COUNT_COLUMNS:
            raise ValueError(f"Unknown count column: {col}")
        out[col] *= float(scale)
    return out


def calibrate_submission(
    df: pd.DataFrame,
    scales: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Load, apply class-wise scales, and return official Kaggle integer format."""
    return finalize_submission_df(apply_scales(load_submission(df), scales))


def blend_submissions(
    dfs: list[pd.DataFrame],
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """Weighted-average compatible submissions by normalized test_id."""
    if not dfs:
        raise ValueError("At least one submission is required")

    if weights is None:
        weights_arr = np.ones(len(dfs), dtype=np.float64)
    else:
        if len(weights) != len(dfs):
            raise ValueError("weights length must match number of submissions")
        weights_arr = np.asarray(weights, dtype=np.float64)

    if not np.isfinite(weights_arr).all() or weights_arr.sum() <= 0:
        raise ValueError("weights must be finite with positive sum")
    weights_arr = weights_arr / weights_arr.sum()

    normalized = [load_submission(df) for df in dfs]
    base_ids = normalized[0][SUBMISSION_ID_COL].tolist()
    base_set = set(base_ids)

    total = pd.DataFrame({SUBMISSION_ID_COL: base_ids})
    total[COUNT_COLUMNS] = 0.0
    total = total.set_index(SUBMISSION_ID_COL)

    for frame, weight in zip(normalized, weights_arr):
        indexed = frame.set_index(SUBMISSION_ID_COL)
        if set(indexed.index) != base_set:
            raise ValueError("All submissions must contain the same test_id values")
        total[COUNT_COLUMNS] += indexed.loc[base_ids, COUNT_COLUMNS] * weight

    return finalize_submission_df(total.reset_index())


def write_submission(df: pd.DataFrame, path: Path | str) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
