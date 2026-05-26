#!/usr/bin/env python3
"""Compare dot_labels.csv totals against train.csv."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dot_cache_analysis import analyze_dot_cache, format_analysis_report, write_analysis


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze dot cache vs train.csv counts.")
    p.add_argument("--data_path", type=str, default="./datasets")
    p.add_argument("--dot_cache", type=str, default="./datasets/dot_labels.csv")
    p.add_argument("--output", type=str, default="./log/dot_cache_analysis.csv")
    p.add_argument(
        "--include_mismatched",
        action="store_true",
        help="Include images listed in MismatchedTrainImages.txt",
    )
    p.add_argument("--rel_err_warn", type=float, default=0.25)
    p.add_argument("--abs_err_warn", type=float, default=5.0)
    args = p.parse_args()

    data_dir = Path(args.data_path)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    dot_cache = Path(args.dot_cache)
    if not dot_cache.is_absolute():
        dot_cache = ROOT / dot_cache
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    if not dot_cache.is_file():
        raise FileNotFoundError(
            f"Missing {dot_cache}. Build it first:\n"
            f"  python -m data.dots --data_path {data_dir} --output {dot_cache}"
        )

    result = analyze_dot_cache(
        data_dir,
        dot_cache,
        exclude_mismatched=not args.include_mismatched,
        rel_err_warn=args.rel_err_warn,
        abs_err_warn=args.abs_err_warn,
    )
    write_analysis(result, output)
    print(format_analysis_report(result))
    print(f"Wrote {output}")
    print(f"Wrote {output.with_name(output.stem + '_summary.csv')}")


if __name__ == "__main__":
    main()
