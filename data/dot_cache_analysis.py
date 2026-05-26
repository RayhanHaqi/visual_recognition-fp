"""Compare dot cache totals against train.csv for label-quality diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data.targets import COUNT_COLUMNS, load_train_counts, list_train_images


@dataclass(frozen=True)
class _CachedDot:
    x: int
    y: int
    class_idx: int


_CLASS_TO_IDX = {name: i for i, name in enumerate(COUNT_COLUMNS)}


def _load_dot_cache_csv(cache_path: Path) -> dict[str, list[_CachedDot]]:
    if not cache_path.is_file():
        return {}
    df = pd.read_csv(cache_path)
    by_image: dict[str, list[_CachedDot]] = {}
    for _, row in df.iterrows():
        class_name = str(row["class"])
        if class_name not in _CLASS_TO_IDX:
            continue
        image_id = str(row["image_id"])
        dot = _CachedDot(int(row["x"]), int(row["y"]), _CLASS_TO_IDX[class_name])
        by_image.setdefault(image_id, []).append(dot)
    return by_image


@dataclass
class DotCacheAnalysisResult:
    per_image: pd.DataFrame
    class_summary: pd.DataFrame
    warnings: dict[str, list[str]] = field(default_factory=dict)
    n_train_images: int = 0
    n_with_dotted: int = 0
    n_missing_dotted: int = 0


def _counts_from_dots(by_image: dict[str, list], image_id: str) -> np.ndarray:
    counts = np.zeros(len(COUNT_COLUMNS), dtype=np.float64)
    for dot in by_image.get(image_id, []):
        counts[dot.class_idx] += 1.0
    return counts


def class_summary_from_rows(per_image: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in COUNT_COLUMNS:
        train_col = f"train_{col}"
        dot_col = f"dot_{col}"
        train_total = float(per_image[train_col].sum()) if train_col in per_image else 0.0
        dot_total = float(per_image[dot_col].sum()) if dot_col in per_image else 0.0
        abs_err = abs(train_total - dot_total)
        rel_err = abs_err / train_total if train_total > 0 else (1.0 if dot_total > 0 else 0.0)
        rows.append(
            {
                "class": col,
                "train_total": train_total,
                "dot_total": dot_total,
                "abs_err": abs_err,
                "rel_err": rel_err,
            }
        )
    return pd.DataFrame(rows).set_index("class")


def analyze_dot_cache(
    data_dir: Path,
    dot_cache_path: Path,
    exclude_mismatched: bool = True,
    rel_err_warn: float = 0.25,
    abs_err_warn: float = 5.0,
) -> DotCacheAnalysisResult:
    """
    Compare dot cache counts to train.csv per image and by class.

    Warnings:
      missing_dotted: Train image has no TrainDotted pair
      no_dots_nonzero_label: dotted pair exists but zero dots with nonzero train counts
      large_mismatch: per-image total count error exceeds thresholds
    """
    counts_df = load_train_counts(data_dir, exclude_mismatched=exclude_mismatched)
    by_image = _load_dot_cache_csv(dot_cache_path)
    train_paths = list_train_images(data_dir, exclude_mismatched=exclude_mismatched)
    train_ids = {p.stem for p in train_paths}

    warnings: dict[str, list[str]] = {
        "missing_dotted": [],
        "no_dots_nonzero_label": [],
        "large_mismatch": [],
    }
    rows: list[dict] = []

    dotted_dir = data_dir / "TrainDotted"
    for image_id in sorted(train_ids):
        has_dotted = (dotted_dir / f"{image_id}.jpg").is_file()
        train_counts = (
            np.array([float(counts_df.loc[image_id, c]) for c in COUNT_COLUMNS], dtype=np.float64)
            if image_id in counts_df.index
            else np.zeros(len(COUNT_COLUMNS), dtype=np.float64)
        )
        dot_counts = _counts_from_dots(by_image, image_id)
        n_dots = int(dot_counts.sum())
        train_total = float(train_counts.sum())
        dot_total = float(dot_counts.sum())
        total_abs_err = abs(train_total - dot_total)
        total_rel_err = total_abs_err / train_total if train_total > 0 else (1.0 if dot_total > 0 else 0.0)

        row: dict = {
            "image_id": image_id,
            "has_dotted": has_dotted,
            "n_dots": n_dots,
            "train_total": train_total,
            "dot_total": dot_total,
            "total_abs_err": total_abs_err,
            "total_rel_err": total_rel_err,
        }
        for i, col in enumerate(COUNT_COLUMNS):
            row[f"train_{col}"] = train_counts[i]
            row[f"dot_{col}"] = dot_counts[i]
            err = abs(train_counts[i] - dot_counts[i])
            rel = err / train_counts[i] if train_counts[i] > 0 else (1.0 if dot_counts[i] > 0 else 0.0)
            row[f"err_{col}"] = err
            row[f"rel_err_{col}"] = rel
        rows.append(row)

        if not has_dotted:
            warnings["missing_dotted"].append(image_id)
        elif n_dots == 0 and train_total > 0:
            warnings["no_dots_nonzero_label"].append(image_id)
        if total_rel_err > rel_err_warn or total_abs_err > abs_err_warn:
            warnings["large_mismatch"].append(image_id)

    per_image = pd.DataFrame(rows)
    class_summary = class_summary_from_rows(per_image)
    return DotCacheAnalysisResult(
        per_image=per_image,
        class_summary=class_summary,
        warnings=warnings,
        n_train_images=len(train_ids),
        n_with_dotted=int(per_image["has_dotted"].sum()),
        n_missing_dotted=int((~per_image["has_dotted"]).sum()),
    )


def format_analysis_report(result: DotCacheAnalysisResult) -> str:
    lines = [
        f"train images={result.n_train_images} | with TrainDotted={result.n_with_dotted} | "
        f"missing TrainDotted={result.n_missing_dotted}",
    ]
    for col in COUNT_COLUMNS:
        row = result.class_summary.loc[col]
        lines.append(
            f"{col}: train={row['train_total']:.0f} dot={row['dot_total']:.0f} "
            f"abs_err={row['abs_err']:.0f} rel_err={row['rel_err']:.3f}"
        )
    for name, ids in result.warnings.items():
        lines.append(f"warn {name}: {len(ids)}")
    return "\n".join(lines)


def dot_cache_gate_failures(
    result: DotCacheAnalysisResult,
    max_class_rel_err: float = 0.10,
    max_large_mismatch: int | None = None,
) -> list[str]:
    failures: list[str] = []
    for col in COUNT_COLUMNS:
        rel_err = float(result.class_summary.loc[col, "rel_err"])
        if rel_err > max_class_rel_err:
            failures.append(f"{col} rel_err {rel_err:.3f} > {max_class_rel_err:.3f}")

    if result.n_missing_dotted:
        failures.append(f"missing TrainDotted pairs: {result.n_missing_dotted}")

    if max_large_mismatch is not None:
        large_mismatch = len(result.warnings.get("large_mismatch", []))
        if large_mismatch > max_large_mismatch:
            failures.append(f"large mismatches {large_mismatch} > {max_large_mismatch}")

    return failures


def write_analysis(result: DotCacheAnalysisResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.per_image.to_csv(output_path, index=False)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    result.class_summary.reset_index().to_csv(summary_path, index=False)
