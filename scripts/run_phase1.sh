#!/bin/bash
# Phase 1 on lab GPU. Control steps with env vars (all default 0 = run step).
#
#   SKIP_INSTALL=1 SKIP_DOWNLOAD=1 SKIP_PREPROCESS=1 SKIP_TESTS=1 \
#   SKIP_SMOKE=1 SKIP_TRAIN=1 SKIP_VALIDATE=1 SKIP_INFER=1 SKIP_SUBMIT=1 \
#   bash scripts/run_phase1.sh 0
#
# FP_CONDA_ENV=selectedtopics_env  GPU id as first argument (default 0)
set -euo pipefail

cd "$(dirname "$0")/.."
GPU="${1:-0}"
RUN_NAME="${RUN_NAME:-baseline}"

# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"
# shellcheck disable=SC1091
source "$(dirname "$0")/kaggle_env.sh"

echo "=================================================="
echo "FP Phase 1 — GPU ${GPU} | env=${CONDA_DEFAULT_ENV:-system}"
echo "=================================================="

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  python setup.py --install
fi

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  python setup.py --download
fi

if [[ "${SKIP_PREPROCESS:-0}" != "1" ]]; then
  python setup.py --preprocess
fi

python setup.py

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  bash scripts/run_tests.sh
fi

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  echo "--- Smoke train (5 epochs) ---"
  python train.py --run_name smoke_v1 --epochs 5 --batch_size 16 --gpu "$GPU" --use_tiles
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "--- Baseline train (30 epochs) ---"
  python train.py --run_name "$RUN_NAME" --epochs 30 --batch_size 16 --gpu "$GPU" --use_tiles
fi

if [[ "${SKIP_VALIDATE:-0}" != "1" && "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "--- Validate (tiled sum RMSE) ---"
  python validate.py "checkpoints/${RUN_NAME}_best.pth" --gpu "$GPU" --shifts 5
fi

if [[ "${SKIP_INFER:-0}" != "1" && "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "--- Inference ---"
  python inference.py "checkpoints/${RUN_NAME}_best.pth" --run_name "$RUN_NAME" --gpu "$GPU" --shifts 5
fi

if [[ "${SKIP_SUBMIT:-0}" != "1" && "${SKIP_INFER:-0}" != "1" ]]; then
  echo "--- Kaggle submit ---"
  bash scripts/submit.sh "submission/${RUN_NAME}.csv" "FP baseline tile regression v1"
fi

echo "Done. Set SKIP_*=1 to resume without repeating completed steps."
