import random
from pathlib import Path


def train_val_split(
    paths: list[Path],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    if val_frac <= 0 or len(paths) < 2:
        return paths, []
    rng = random.Random(seed)
    shuffled = paths.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    val_paths = sorted(shuffled[:n_val], key=lambda p: p.stem)
    train_paths = sorted(shuffled[n_val:], key=lambda p: p.stem)
    return train_paths, val_paths
