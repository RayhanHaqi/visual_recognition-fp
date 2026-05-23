"""Ground-truth count loading from train.csv."""

from pathlib import Path

import pandas as pd

COUNT_COLUMNS = [
    "adult_males",
    "adult_females",
    "subadult_males",
    "subadult_females",
    "pups",
]


def image_id_from_path(path: Path | str) -> str:
    """Competition id: filename without extension."""
    return Path(path).stem


def load_train_counts(data_dir: Path, exclude_mismatched: bool = True) -> pd.DataFrame:
    """
    Load train.csv indexed by image id (column 'id' or first column).
    Returns DataFrame with COUNT_COLUMNS.
    """
    csv_path = data_dir / "train.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    if "id" in df.columns:
        df = df.set_index("id", drop=False)
    else:
        df = df.rename(columns={df.columns[0]: "id"}).set_index("id", drop=False)

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


def counts_for_image(df: pd.DataFrame, image_id: str):
    import numpy as np

    row = df.loc[image_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return np.array([float(row[c]) for c in COUNT_COLUMNS], dtype="float32")


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
