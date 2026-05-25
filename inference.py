#!/usr/bin/env python3
"""Generate Kaggle submission CSV from test images."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.predict import predict_image_tiled
from data.targets import COUNT_COLUMNS, list_test_images
from model.build import build_counter
from utils.io import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--run_name", type=str, default="submission")
    p.add_argument("--data_path", type=str, default="./datasets")
    p.add_argument("--test_subdir", type=str, default="Test",
                   help="Test or Test_scaled after preprocess")
    p.add_argument("--tile_size", type=int, default=None)
    p.add_argument("--shifts", type=int, default=5)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--pup_scale", type=float, default=1.0,
                   help="Optional post-process: scale pup counts")
    p.add_argument("--output", type=str, default=None)
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

    data_dir = Path(args.data_path)
    sample_path = data_dir / "sample_submission.csv"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Missing {sample_path}")
    test_paths = list_test_images(data_dir, test_subdir=args.test_subdir)
    path_by_id = {p.stem: p for p in test_paths}
    path_by_id.update({p.name: p for p in test_paths})

    sample = pd.read_csv(sample_path)
    id_col = "id" if "id" in sample.columns else sample.columns[0]
    n_sample = len(sample)
    n_test = len(test_paths)
    if n_sample > n_test + 10:
        raise ValueError(
            f"sample_submission.csv has {n_sample} rows but {args.test_subdir}/ has "
            f"{n_test} images. Rebuild with:\n"
            f"  python -m data.fix_sample_submission --data_path {args.data_path} --limit 100"
        )

    rows = []
    for img_id in tqdm(sample[id_col].astype(str), desc="inference"):
        stem = Path(img_id).stem
        path = path_by_id.get(stem) or path_by_id.get(img_id)
        if path is None:
            raise FileNotFoundError(f"Test image not found for id={img_id}")
        pred = predict_image_tiled(
            model, path, tile_size, device,
            shifts=args.shifts, stride=stride, batch_size=args.batch_size,
        )
        if args.pup_scale != 1.0:
            pred = pred.copy()
            pred[4] *= args.pup_scale
        row = {id_col: img_id}
        for i, col in enumerate(COUNT_COLUMNS):
            row[col] = max(0.0, float(pred[i]))
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_dir = ROOT / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"{args.run_name}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
