import csv
from pathlib import Path

import torch


def next_run_id(tracker_path: Path) -> int:
    if tracker_path.is_file():
        run_id = int(tracker_path.read_text().strip()) + 1
    else:
        run_id = 1
    tracker_path.write_text(str(run_id))
    return run_id


def init_log_csv(log_path: Path, fieldnames: list[str]):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.is_file():
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_log_row(log_path: Path, row: dict, fieldnames: list[str]):
    with open(log_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_rmse: float, args_dict: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "best_rmse": best_rmse,
            "args": args_dict,
        },
        path,
    )


def load_checkpoint(path: Path, model, device, optimizer=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
