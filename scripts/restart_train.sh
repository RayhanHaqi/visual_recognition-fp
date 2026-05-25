#!/bin/bash
# Stop stale FP jobs and start a fresh training run (train only, no inference).
set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source scripts/conda_env.sh

BACKBONE=${1:-resnet50}
EPOCHS=${2:-30}
BS=${3:-128}
LR=${4:-1e-4}
GPU=${5:-0}
TILE=${6:-299}
WORKERS=${7:-0}
RUN_SUFFIX=${8:-v2}

RUN_NAME="fp_${BACKBONE}_e${EPOCHS}_bs${BS}_t${TILE}_${RUN_SUFFIX}"

echo "Stopping stale train.py jobs..."
pkill -f 'train.py.*smoke_v1' 2>/dev/null || true
pkill -f "train.py.*fp_${BACKBONE}" 2>/dev/null || true
sleep 2
pgrep -af 'train.py' || echo "(no train.py running)"

echo "=================================================="
echo "FP restart: $RUN_NAME | GPU $GPU | workers=$WORKERS"
echo "=================================================="

python train.py \
  --run_name "$RUN_NAME" \
  --backbone "$BACKBONE" \
  --epochs "$EPOCHS" \
  --batch_size "$BS" \
  --lr "$LR" \
  --tile_size "$TILE" \
  --gpu "$GPU" \
  --workers "$WORKERS" \
  --val_shifts 1 \
  --use_tiles

echo "Done. Best: checkpoints/${RUN_NAME}_best.pth"
echo "Log: log/${RUN_NAME}.csv"
echo ""
echo "=================================================="
echo "NEXT: run the rest of the pipeline (inference → submit)"
echo "=================================================="
echo "python inference.py checkpoints/${RUN_NAME}_best.pth \\"
echo "  --run_name ${RUN_NAME} --gpu ${GPU} --shifts 5 --stride ${TILE}"
echo ""
echo "source scripts/kaggle_env.sh"
echo "bash scripts/submit.sh submission/${RUN_NAME}.csv \"FP ${RUN_NAME}\""
echo ""
echo "Optional: bash scripts/push_progress.sh ${RUN_NAME}"
