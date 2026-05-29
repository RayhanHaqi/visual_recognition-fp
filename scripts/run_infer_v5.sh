#!/bin/bash
# Standard fast inference for a trained checkpoint (tmux-friendly).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"
# shellcheck disable=SC1091
source "$(dirname "$0")/lib/fp_paths.sh"

PHASE_TITLE=${PHASE_TITLE:-inference}

echo "=================================================="
echo "FP ${PHASE_TITLE} — infer"
echo "=================================================="

CKPT=${CKPT:-checkpoints/fp_resnet50_e20_bs256_t299_balanced_dots_v5_best.pth}
RUN_NAME=${RUN_NAME:-balanced_dots_v5}
GPU=${GPU:-1}
TEST_SUBDIR=${TEST_SUBDIR:-Test_scaled_0.5}
BATCH_SIZE=${BATCH_SIZE:-512}
SHIFTS=${SHIFTS:-5}
AMP=${AMP:-0}
# Optional override; default: inference.py uses checkpoint tile_size for stride
STRIDE=${STRIDE:-}

fp_require_checkpoint "$CKPT"
fp_require_test_subdir "$TEST_SUBDIR"
mkdir -p log submission

AMP_ARGS=()
if [[ "$AMP" == "1" ]]; then
  AMP_ARGS+=(--amp)
fi

STRIDE_ARGS=()
if [[ -n "$STRIDE" ]]; then
  STRIDE_ARGS+=(--stride "$STRIDE")
fi

python inference.py "$CKPT" \
  --run_name "$RUN_NAME" \
  --gpu "$GPU" \
  --test_subdir "$TEST_SUBDIR" \
  --batch_size "$BATCH_SIZE" \
  --shifts "$SHIFTS" \
  "${STRIDE_ARGS[@]}" \
  "${AMP_ARGS[@]}" \
  2>&1 | tee "log/${RUN_NAME}_infer.log"
