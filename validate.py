#!/usr/bin/env python3
"""Validate checkpoint with tile aggregation on held-out train images."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_train_val_paths
from data.targets import counts_for_image
from data.tiling import aggregate_tile_predictions, iter_tile_windows
from model.build import build_counter
from utils.io import load_checkpoint
from utils.metrics import rmse_numpy


def predict_image_tiled(model, image_path: Path, tile_size: int, device, shifts: int, stride: int):
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    preds, windows = [], []
    model.eval()
    with torch.no_grad():
        for win in iter_tile_windows(w, h, tile_size, stride=stride, shifts=shifts):
            crop = img[win.y0 : win.y1, win.x0 : win.x1]
            pil = Image.fromarray(crop).resize((tile_size, tile_size), Image.BILINEAR)
            t = normalize(T.ToTensor()(pil)).unsqueeze(0).to(device)
            pred = torch.clamp(model(t), min=0).cpu().numpy()[0]
            preds.append(pred)
            windows.append(win)
    if not preds:
        pil = Image.fromarray(img).resize((tile_size, tile_size), Image.BILINEAR)
        t = normalize(T.ToTensor()(pil)).unsqueeze(0).to(device)
        return torch.clamp(model(t), min=0).cpu().numpy()[0]
    return aggregate_tile_predictions(preds, windows, w, h, mode="area_weighted")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--data_path", type=str, default="./datasets")
    p.add_argument("--tile_size", type=int, default=299)
    p.add_argument("--shifts", type=int, default=3)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    backbone = ckpt.get("args", {}).get("backbone", "resnet50")
    model = build_counter(backbone, pretrained=False).to(device)
    load_checkpoint(Path(args.checkpoint), model, device)

    _, val_paths, counts_df = build_train_val_paths(
        Path(args.data_path), val_frac=args.val_frac, seed=args.seed
    )
    stride = args.stride or args.tile_size // 2
    pred_rows, tgt_rows = [], []
    for path in tqdm(val_paths, desc="val"):
        pred = predict_image_tiled(
            model, path, args.tile_size, device, args.shifts, stride
        )
        tgt = counts_for_image(counts_df, path.stem)
        pred_rows.append(pred)
        tgt_rows.append(tgt)

    rmse = rmse_numpy(np.stack(pred_rows), np.stack(tgt_rows))
    print(f"Val RMSE (tiled): {rmse:.4f} on {len(val_paths)} images")


if __name__ == "__main__":
    main()
