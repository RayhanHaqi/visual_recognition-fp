#!/bin/bash
# Phase 1 on lab GPU: setup → preprocess → test → smoke train → baseline → infer → submit
# Usage: bash scripts/run_phase1.sh [GPU_ID]
set -euo pipefail

cd "$(dirname "$0")/.."
GPU="${1:-0}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "visualrecognition" ]]; then
  if command -v conda &>/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate visualrecognition 2>/dev/null || true
  fi
fi

echo "=================================================="
echo "FP Phase 1 — GPU ${GPU}"
echo "=================================================="

python setup.py --install
python setup.py --download
python setup.py --preprocess
python setup.py

bash scripts/run_tests.sh

echo "--- Smoke train (5 epochs) ---"
python train.py --run_name smoke_v1 --epochs 5 --batch_size 16 --gpu "$GPU" --use_tiles

echo "--- Baseline train (30 epochs) ---"
python train.py --run_name baseline --epochs 30 --batch_size 16 --gpu "$GPU" --use_tiles

echo "--- Validate (tiled val RMSE) ---"
python validate.py checkpoints/baseline_best.pth --gpu "$GPU" --shifts 5

echo "--- Inference ---"
python inference.py checkpoints/baseline_best.pth --run_name baseline --gpu "$GPU" --shifts 5

echo "--- Kaggle submit ---"
bash scripts/submit.sh submission/baseline.csv "FP baseline tile regression v1"

echo "Done. Archive leaderboard screenshot for report (due 2026/05/31)."
