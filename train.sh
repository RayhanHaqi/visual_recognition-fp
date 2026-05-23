#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Prefer course conda env when available
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "visualrecognition" ]]; then
  if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate visualrecognition 2>/dev/null || true
  fi
fi

BACKBONE=${1:-resnet50}
EPOCHS=${2:-30}
BS=${3:-16}
LR=${4:-1e-4}
GPU=${5:-0}
TILE=${6:-299}

RUN_NAME="fp_${BACKBONE}_e${EPOCHS}_bs${BS}_t${TILE}"

echo "=================================================="
echo "FP train pipeline: $RUN_NAME"
echo "=================================================="

python train.py \
  --run_name "$RUN_NAME" \
  --backbone "$BACKBONE" \
  --epochs "$EPOCHS" \
  --batch_size "$BS" \
  --lr "$LR" \
  --tile_size "$TILE" \
  --gpu "$GPU" \
  --use_tiles

echo "Inference..."
python inference.py "checkpoints/${RUN_NAME}_best.pth" \
  --run_name "$RUN_NAME" \
  --gpu "$GPU" \
  --shifts 5

echo "Done. Submission: submission/${RUN_NAME}.csv"
