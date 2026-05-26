from data.targets import COUNT_COLUMNS, image_id_from_path, load_train_counts
from data.tiling import aggregate_tile_predictions, iter_tile_windows

__all__ = [
    "COUNT_COLUMNS",
    "load_train_counts",
    "image_id_from_path",
    "SeaLionImageDataset",
    "SeaLionTileDataset",
    "iter_tile_windows",
    "aggregate_tile_predictions",
]


def __getattr__(name):
    if name in {"SeaLionImageDataset", "SeaLionTileDataset"}:
        from data.dataset import SeaLionImageDataset, SeaLionTileDataset

        return {
            "SeaLionImageDataset": SeaLionImageDataset,
            "SeaLionTileDataset": SeaLionTileDataset,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
