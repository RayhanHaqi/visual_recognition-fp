"""Ground-truth count loading from train.csv and Kaggle submission schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Official Kaggle column order (train labels + submission).
COUNT_COLUMNS = [
    "adult_males",
    "subadult_males",
    "adult_females",
    "juveniles",
    "pups",
]

SUBMISSION_ID_COL = "test_id"
TRAIN_ID_COL = "train_id"

# Checkpoints saved before May 2026 column-order fix used this layout.
LEGACY_COUNT_COLUMNS = [
    "adult_males",
    "adult_females",
    "subadult_males",
    "subadult_females",
    "pups",
]

SUBMISSION_COLUMNS = [SUBMISSION_ID_COL, *COUNT_COLUMNS]


def image_id_from_path(path: Path | str) -> str:
    """Competition id: filename without extension."""
    return Path(path).stem


def normalize_test_id(value) -> int | str:
    stem = Path(str(value)).stem
    return int(stem) if stem.isdigit() else stem


def submission_id_column(df: pd.DataFrame) -> str:
    if SUBMISSION_ID_COL in df.columns:
        return SUBMISSION_ID_COL
    if "id" in df.columns:
        return "id"
    return df.columns[0]


def count_columns_from_checkpoint(ckpt_args: dict | None) -> list[str]:
    if not ckpt_args:
        return LEGACY_COUNT_COLUMNS
    saved = ckpt_args.get("count_columns")
    if saved:
        return list(saved)
    return LEGACY_COUNT_COLUMNS


def pred_vector_to_submission_row(
    pred,
    source_columns: list[str] | None = None,
) -> dict[str, float]:
    """Map model output (5-d) to Kaggle submission count columns."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if len(pred) != 5:
        raise ValueError(f"Expected 5-d prediction, got shape {pred.shape}")

    source_columns = source_columns or COUNT_COLUMNS
    if source_columns == LEGACY_COUNT_COLUMNS:
        values = {
            "adult_males": pred[0],
            "subadult_males": pred[2],
            "adult_females": pred[1],
            "juveniles": pred[3],
            "pups": pred[4],
        }
    elif source_columns == COUNT_COLUMNS:
        values = {col: pred[i] for i, col in enumerate(COUNT_COLUMNS)}
    else:
        pred_map = {col: float(pred[i]) for i, col in enumerate(source_columns)}
        values = {col: pred_map[col] for col in COUNT_COLUMNS}

    return {col: max(0.0, float(values[col])) for col in COUNT_COLUMNS}


def finalize_submission_df(df: pd.DataFrame) -> pd.DataFrame:
    """Kaggle requires non-negative integer counts."""
    out = df[SUBMISSION_COLUMNS].copy()
    out[SUBMISSION_ID_COL] = out[SUBMISSION_ID_COL].map(normalize_test_id)
    for col in COUNT_COLUMNS:
        out[col] = out[col].clip(lower=0).round().astype(int)
    return out


def _normalize_train_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TRAIN_ID_COL in df.columns:
        df = df.rename(columns={TRAIN_ID_COL: "id"})
    elif "id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "id"})

    if "subadult_females" in df.columns and "juveniles" not in df.columns:
        df = df.rename(columns={"subadult_females": "juveniles"})

    df["id"] = df["id"].astype(str)
    return df.set_index("id", drop=False)


def load_train_counts(data_dir: Path, exclude_mismatched: bool = True) -> pd.DataFrame:
    """
    Load train.csv indexed by image id (train_id or id).
    Returns DataFrame with COUNT_COLUMNS.
    """
    csv_path = data_dir / "train.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = _normalize_train_columns(pd.read_csv(csv_path))

    missing = [c for c in COUNT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"train.csv missing columns: {missing}")

    if exclude_mismatched:
        bad = load_mismatched_ids(data_dir)
        if bad:
            df = df[~df.index.isin(bad)]

    return df


def load_mismatched_ids(data_dir: Path) -> set[str]:
    path = data_dir / "MismatchedTrainImages.txt"
    if not path.is_file():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ids.add(Path(line).stem)
    return ids


def counts_for_image(df: pd.DataFrame, image_id: str) -> np.ndarray:
    row = df.loc[image_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return np.array([float(row[c]) for c in COUNT_COLUMNS], dtype=np.float32)


def list_train_images(data_dir: Path, exclude_mismatched: bool = True) -> list[Path]:
    train_dir = data_dir / "Train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing {train_dir}")
    bad = load_mismatched_ids(data_dir) if exclude_mismatched else set()
    paths = []
    for p in sorted(train_dir.glob("*.jpg")):
        if p.stem not in bad:
            paths.append(p)
    return paths


def list_test_images(data_dir: Path, test_subdir: str = "Test") -> list[Path]:
    test_dir = data_dir / test_subdir
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing {test_dir}")
    return sorted(test_dir.glob("*.jpg"))
