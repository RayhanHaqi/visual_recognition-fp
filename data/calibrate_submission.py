"""CLI for class scaling and weighted blending of submission CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.submission_ops import apply_scales, blend_submissions, load_submission, write_submission

ROOT = Path(__file__).resolve().parent.parent


def _parse_scale(values: list[str] | None) -> dict[str, float]:
    scales: dict[str, float] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Scale must be class=factor, got: {item}")
        col, value = item.split("=", 1)
        scales[col] = float(value)
    return scales


def _parse_weights(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(part) for part in value.split(",") if part]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Apply class scales and optional weighted blending to submission CSVs."
    )
    p.add_argument("input_csv", type=str)
    p.add_argument("--blend", nargs="*", default=[], help="Additional CSVs to blend with input_csv")
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Comma-separated weights for input_csv plus --blend CSVs, e.g. 0.7,0.3",
    )
    p.add_argument(
        "--scale",
        action="append",
        default=[],
        help="Class multiplier, repeatable: --scale pups=1.2",
    )
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    input_paths = [Path(args.input_csv), *[Path(path) for path in args.blend]]
    frames = [
        pd.read_csv(path if path.is_absolute() else ROOT / path)
        for path in input_paths
    ]

    if len(frames) > 1:
        out = blend_submissions(frames, weights=_parse_weights(args.weights))
    else:
        out = load_submission(frames[0])

    scales = _parse_scale(args.scale)
    if scales:
        from data.targets import finalize_submission_df

        out = finalize_submission_df(apply_scales(out, scales))

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    write_submission(out, out_path)
    print(f"Wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
