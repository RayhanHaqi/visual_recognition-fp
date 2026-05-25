#!/usr/bin/env python3
"""Validate checkpoint with tiled inference (same contract as test submission)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_train_val_paths
from data.predict import predict_image_tiled
from data.targets import counts_for_image
from model.build import build_counter
from utils.io import load_checkpoint
from utils.metrics import rmse_numpy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--data_path", type=str, default="./datasets")
    p.add_argument("--tile_size", type=int, default=None)
    p.add_argument("--shifts", type=int, default=5)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    backbone = ckpt_args.get("backbone", "resnet50")
    tile_size = args.tile_size or ckpt_args.get("tile_size", 299)
    stride = args.stride if args.stride is not None else tile_size

    model = build_counter(backbone, pretrained=False).to(device)
    load_checkpoint(ckpt_path, model, device)

    _, val_paths, counts_df = build_train_val_paths(
        Path(args.data_path), val_frac=args.val_frac, seed=args.seed
    )
    pred_rows, tgt_rows = [], []
    for path in tqdm(val_paths, desc="val"):
        pred = predict_image_tiled(
            model, path, tile_size, device,
            shifts=args.shifts, stride=stride, batch_size=args.batch_size,
        )
        tgt = counts_for_image(counts_df, path.stem)
        pred_rows.append(pred)
        tgt_rows.append(tgt)

    rmse = rmse_numpy(np.stack(pred_rows), np.stack(tgt_rows))
    print(f"Val RMSE (tiled sum): {rmse:.4f} on {len(val_paths)} images")


if __name__ == "__main__":
    main()
