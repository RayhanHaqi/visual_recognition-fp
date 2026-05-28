#!/usr/bin/env python3
"""Generate Kaggle submission CSV from test images."""

import argparse
import sys
import time
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
from model.build import build_counter_from_checkpoint_args
from utils.io import load_checkpoint
from utils.timefmt import format_duration

_INFER_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)


def _resolve_test_path(path_by_id: dict, img_id: str) -> Path | None:
    stem = Path(str(img_id)).stem
    return path_by_id.get(stem) or path_by_id.get(str(img_id)) or path_by_id.get(f"{stem}.jpg")


def _plan_model_runs(
    sample_ids: list[str],
    path_by_id: dict,
    zero_counts: dict[str, float],
    max_images: int | None,
) -> tuple[list[tuple[str, Path]], dict[str | int, dict[str, float]], int]:
    """Unique test ids that need a model forward pass vs zero-fill cache entries."""
    predict_list: list[tuple[str, Path]] = []
    pred_cache: dict[str | int, dict[str, float]] = {}
    n_zero = 0
    seen: set[str | int] = set()

    for img_id in sample_ids:
        key = normalize_test_id(img_id)
        if key in seen:
            continue
        seen.add(key)
        path = _resolve_test_path(path_by_id, img_id)
        if path is None:
            pred_cache[key] = zero_counts.copy()
            n_zero += 1
        elif max_images is not None and len(predict_list) >= max_images:
            pred_cache[key] = zero_counts.copy()
            n_zero += 1
        else:
            predict_list.append((key, path))

    return predict_list, pred_cache, n_zero


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
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Profile/debug: run model on at most N unique test images (still writes full sample rows)",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="CUDA autocast for model forward (may speed GPU; test with --profile_inference first)",
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

    model = build_counter_from_checkpoint_args(ckpt_args, pretrained=False).to(device)
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

    zero_counts = {col: 0.0 for col in COUNT_COLUMNS}
    sample_ids = sample[id_col].astype(str).tolist()
    predict_list, pred_cache, n_zero = _plan_model_runs(
        sample_ids, path_by_id, zero_counts, args.max_images
    )
    profile = InferenceTimings() if args.profile_inference else None

    n_model = len(predict_list)
    print(
        f"Model runs planned: {n_model} | "
        f"zero-fill ids: {n_zero} | "
        f"submission rows: {len(sample_ids)}"
    )

    run_t0 = time.perf_counter()
    with tqdm(
        total=n_model,
        desc="inference",
        unit="img",
        bar_format=_INFER_BAR_FORMAT,
    ) as pbar:
        for key, path in predict_list:
            pred = predict_image_tiled(
                model, path, tile_size, device,
                shifts=args.shifts, stride=stride, batch_size=args.batch_size,
                timings=profile,
                use_amp=args.amp,
            )
            counts = pred_vector_to_submission_row(pred, source_columns=source_columns)
            if args.pup_scale != 1.0:
                counts["pups"] *= args.pup_scale
            pred_cache[key] = counts
            pbar.update(1)

    total_elapsed = time.perf_counter() - run_t0
    if n_model:
        imgs_per_sec = n_model / total_elapsed
        extra = ""
        full_unique = len({normalize_test_id(i) for i in sample_ids})
        if full_unique > n_model and imgs_per_sec > 0:
            extra = f" | est. full {full_unique} imgs ~{format_duration(full_unique / imgs_per_sec)}"
        print(
            f"Inference elapsed: {format_duration(total_elapsed)} "
            f"({imgs_per_sec:.3f} img/s){extra}"
        )

    rows = []
    for img_id in sample_ids:
        key = normalize_test_id(img_id)
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
