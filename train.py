#!/usr/bin/env python3
"""Train tile/image regression model for sea lion counts."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (
    SeaLionImageDataset,
    SeaLionTileDataset,
    build_train_val_paths,
)
from data.predict import predict_image_tiled
from data.targets import COUNT_COLUMNS
from model.build import build_counter
from model.losses import rmse_loss
from utils.io import append_log_row, init_log_csv, next_run_id, save_checkpoint
from utils.metrics import rmse_numpy
from utils.seed import set_seed


LOG_FIELDS = ["epoch", "train_loss", "train_rmse", "val_rmse", "lr", "best_rmse"]


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--data_path", type=str, default="./datasets")
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument("--tile_size", type=int, default=299)
    p.add_argument("--tiles_per_image", type=int, default=8)
    p.add_argument("--use_tiles", action="store_true", default=True)
    p.add_argument("--no_tiles", dest="use_tiles", action="store_false")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers (use spawn if >0 on CUDA; try 4 on lab machine)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    p.add_argument("--save_path", type=str, default="./checkpoints")
    p.add_argument("--log_path", type=str, default="./log")
    p.add_argument("--pct_start", type=float, default=0.1)
    p.add_argument(
        "--val_shifts",
        type=int,
        default=1,
        help="Shifts for in-training validation (1 = fast; inference may use 5)",
    )
    p.add_argument(
        "--val_stride",
        type=int,
        default=None,
        help="Val tile stride (default: tile_size = non-overlapping sum grid)",
    )
    p.add_argument(
        "--label_mode",
        type=str,
        default="area",
        choices=["area", "dots"],
        help="Tile target: area-fraction of image counts or TrainDotted dot counts",
    )
    p.add_argument(
        "--dot_cache",
        type=str,
        default="./datasets/dot_labels.csv",
        help="CSV cache from python -m data.dots",
    )
    p.add_argument(
        "--build_dot_cache",
        action="store_true",
        help="Build dot cache before training if missing",
    )
    return p.parse_args()


@torch.no_grad()
def evaluate_images(
    model,
    val_paths,
    counts_df,
    tile_size,
    device,
    shifts=3,
    stride=None,
    batch_size=8,
):
    """Tiled aggregation for validation during training."""
    if not val_paths:
        return float("nan")
    from data.targets import counts_for_image

    stride = stride if stride is not None else tile_size
    preds, tgts = [], []
    for path in tqdm(val_paths, desc="val", leave=False):
        preds.append(
            predict_image_tiled(
                model, path, tile_size, device,
                shifts=shifts, stride=stride, batch_size=batch_size,
            )
        )
        tgts.append(counts_for_image(counts_df, path.stem))
    return rmse_numpy(np.stack(preds), np.stack(tgts))


def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, use_amp):
    model.train()
    total_loss = 0.0
    all_pred, all_tgt = [], []
    for images, targets, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp and device.type == "cuda"):
            pred = model(images)
            loss = rmse_loss(pred, targets)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        all_pred.append(torch.clamp(pred.detach(), min=0).cpu().numpy())
        all_tgt.append(targets.cpu().numpy())
    pred = np.concatenate(all_pred, axis=0)
    tgt = np.concatenate(all_tgt, axis=0)
    return total_loss / len(loader), rmse_numpy(pred, tgt)


def main():
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_path)

    if args.run_name is None:
        rid = next_run_id(ROOT / "run_tracker.txt")
        args.run_name = f"run{rid}"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Run: {args.run_name} | Device: {device}")

    train_paths, val_paths, counts_df = build_train_val_paths(
        data_dir, val_frac=args.val_frac, seed=args.seed
    )
    print(f"Train images: {len(train_paths)}, Val images: {len(val_paths)}")

    dot_cache_path = None
    dots_by_image = None
    if args.use_tiles and args.label_mode == "dots":
        from data.dots import build_dot_cache, load_dot_cache

        dot_cache_path = Path(args.dot_cache)
        if not dot_cache_path.is_absolute():
            dot_cache_path = ROOT / dot_cache_path
        if not dot_cache_path.is_file() or args.build_dot_cache:
            if not dot_cache_path.is_file():
                print(f"Building dot cache -> {dot_cache_path}")
            else:
                print(f"Rebuilding dot cache -> {dot_cache_path}")
            build_dot_cache(data_dir, dot_cache_path)
        dots_by_image = load_dot_cache(dot_cache_path)
        n_dots = sum(len(v) for v in dots_by_image.values())
        print(f"Loaded dot cache: {len(dots_by_image)} images, {n_dots} dots")

    if args.use_tiles:
        train_ds = SeaLionTileDataset(
            train_paths, counts_df,
            tile_size=args.tile_size,
            tiles_per_image=args.tiles_per_image,
            train=True,
            label_mode=args.label_mode,
            dots_by_image=dots_by_image,
        )
    else:
        train_ds = SeaLionImageDataset(
            train_paths, counts_df, tile_size=args.tile_size, train=True
        )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    if args.workers > 0 and device.type == "cuda":
        loader_kwargs["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(train_ds, **loader_kwargs)

    model = build_counter(args.backbone, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(1, steps), pct_start=args.pct_start
    )
    scaler = GradScaler("cuda") if args.amp and device.type == "cuda" else None

    save_dir = Path(args.save_path)
    log_path = Path(args.log_path) / f"{args.run_name}.csv"
    init_log_csv(log_path, LOG_FIELDS)
    best_path = save_dir / f"{args.run_name}_best.pth"
    best_rmse = float("inf")
    args_dict = vars(args)
    args_dict["count_columns"] = COUNT_COLUMNS
    if args.label_mode == "dots":
        args_dict["dot_cache"] = str(dot_cache_path)

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()

        train_t0 = time.perf_counter()
        train_loss, train_rmse = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, args.amp
        )
        train_secs = time.perf_counter() - train_t0

        val_t0 = time.perf_counter()
        val_rmse = evaluate_images(
            model,
            val_paths,
            counts_df,
            args.tile_size,
            device,
            shifts=args.val_shifts,
            stride=args.val_stride if args.val_stride is not None else args.tile_size,
            batch_size=args.batch_size,
        )
        val_secs = time.perf_counter() - val_t0
        epoch_secs = time.perf_counter() - epoch_t0

        lr = optimizer.param_groups[0]["lr"]
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            save_checkpoint(best_path, model, optimizer, epoch, best_rmse, args_dict)
            print(f"  Saved best checkpoint (val_rmse={best_rmse:.4f})")

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_rmse": round(train_rmse, 4),
            "val_rmse": round(val_rmse, 4) if not np.isnan(val_rmse) else "",
            "lr": lr,
            "best_rmse": round(best_rmse, 4),
        }
        append_log_row(log_path, row, LOG_FIELDS)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train {format_duration(train_secs)} | val {format_duration(val_secs)} | "
            f"total {format_duration(epoch_secs)} | "
            f"train_rmse={train_rmse:.4f} | val_rmse={val_rmse:.4f} | best={best_rmse:.4f}"
        )

    last_path = save_dir / f"{args.run_name}_last.pth"
    save_checkpoint(last_path, model, optimizer, args.epochs, best_rmse, args_dict)
    print(f"Done. Best: {best_path}")


if __name__ == "__main__":
    main()
