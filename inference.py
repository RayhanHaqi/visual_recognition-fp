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

from data.predict import InferenceTimings, format_inference_profile, predict_image_tiled
from data.targets import (
    COUNT_COLUMNS,
    SUBMISSION_ID_COL,
    count_columns_from_checkpoint,
    finalize_submission_df,
    list_test_images,
    normalize_test_id,
    pred_vector_to_submission_row,
    submission_id_column,
)
from model.build import build_counter
from utils.io import load_checkpoint


def _resolve_test_path(path_by_id: dict, img_id: str) -> Path | None:
    stem = Path(str(img_id)).stem
    return path_by_id.get(stem) or path_by_id.get(str(img_id)) or path_by_id.get(f"{stem}.jpg")


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
    p.add_argument(
        "--profile_inference",
        action="store_true",
        help="Collect per-stage inference timings and write log/{run_name}_infer_profile.csv",
    )
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    backbone = ckpt_args.get("backbone", "resnet50")
    tile_size = args.tile_size or ckpt_args.get("tile_size", 299)
    stride = args.stride if args.stride is not None else tile_size
    source_columns = count_columns_from_checkpoint(ckpt_args)

    model = build_counter(backbone, pretrained=False).to(device)
    load_checkpoint(ckpt_path, model, device)
    model.eval()

    data_dir = Path(args.data_path)
    sample_path = data_dir / "sample_submission.csv"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Missing {sample_path}")
    test_paths = list_test_images(data_dir, test_subdir=args.test_subdir)
    path_by_id = {p.stem: p for p in test_paths}
    path_by_id.update({p.name: p for p in test_paths})

    sample = pd.read_csv(sample_path)
    id_col = submission_id_column(sample)
    n_sample = len(sample)
    n_test = len(test_paths)
    print(f"sample_submission rows: {n_sample} | {args.test_subdir} images: {n_test}")

    pred_cache: dict[str | int, dict[str, float]] = {}
    zero_counts = {col: 0.0 for col in COUNT_COLUMNS}
    rows = []
    n_model = 0
    n_zero = 0
    profile = InferenceTimings() if args.profile_inference else None

    for img_id in tqdm(sample[id_col].astype(str), desc="inference"):
        key = normalize_test_id(img_id)
        if key not in pred_cache:
            path = _resolve_test_path(path_by_id, img_id)
            if path is None:
                pred_cache[key] = zero_counts.copy()
                n_zero += 1
            else:
                pred = predict_image_tiled(
                    model, path, tile_size, device,
                    shifts=args.shifts, stride=stride, batch_size=args.batch_size,
                    timings=profile,
                )
                counts = pred_vector_to_submission_row(pred, source_columns=source_columns)
                if args.pup_scale != 1.0:
                    counts["pups"] *= args.pup_scale
                pred_cache[key] = counts
                n_model += 1

        row = {SUBMISSION_ID_COL: key}
        row.update(pred_cache[key])
        rows.append(row)

    out_df = finalize_submission_df(pd.DataFrame(rows))
    out_dir = ROOT / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"{args.run_name}.csv"
    out_df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} ({len(out_df)} rows) | "
        f"model runs={n_model} | zero-fill ids={n_zero}"
    )

    if profile is not None:
        summary = format_inference_profile(profile)
        log_dir = ROOT / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        profile_path = log_dir / f"{args.run_name}_infer_profile.csv"
        pd.DataFrame([summary]).to_csv(profile_path, index=False)
        print(
            "Inference profile | "
            f"images={int(summary['n_images'])} "
            f"tiles={int(summary['n_tiles'])} "
            f"batch_forwards={int(summary['n_batch_forwards'])} | "
            f"{summary['images_per_sec']:.3f} img/s "
            f"{summary['tiles_per_sec']:.1f} tile/s | "
            f"load={summary['load_pct']:.1f}% "
            f"windows={summary['windows_pct']:.1f}% "
            f"preprocess={summary['preprocess_pct']:.1f}% "
            f"h2d={summary['h2d_pct']:.1f}% "
            f"forward={summary['forward_pct']:.1f}% "
            f"aggregate={summary['aggregate_pct']:.1f}%"
        )
        print(f"Wrote {profile_path}")


if __name__ == "__main__":
    main()
