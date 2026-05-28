#!/bin/bash
# Train v9 then v10, infer, pup×1.2, submit each (tmux-friendly).
#
#   tmux new -s fp_top3_train
#   bash scripts/run_top3_train_queue.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"

GPU=${GPU:-1}
BEST_RMSE=${BEST_RMSE:-14.57729}
INFER_BS=${INFER_BS:-256}
TEST_SUBDIR=${TEST_SUBDIR:-Test_scaled_0.5}

wait_for_gpu() {
  while pgrep -f 'train.py|inference.py' >/dev/null 2>&1; do
    pgrep -af 'train.py|inference.py' 2>/dev/null || true
    sleep 60
  done
}

run_one() {
  local train_script=$1
  local ckpt=$2
  local run=$3
  local msg=$4

  bash "$train_script"
  [[ -f "$ckpt" ]] || { echo "Missing $ckpt"; exit 1; }

  PHASE_TITLE="infer $run" \
    CKPT="$ckpt" RUN_NAME="$run" GPU="$GPU" \
    TEST_SUBDIR="$TEST_SUBDIR" BATCH_SIZE="$INFER_BS" AMP=1 \
    bash scripts/run_infer_v5.sh

  RUN_NAME="$run" SUBMIT_MSG="$msg" BEST_RMSE="$BEST_RMSE" \
    bash scripts/finish_phase_run.sh
}

echo "=== Top-3 train queue (target < ${BEST_RMSE}) ==="
wait_for_gpu

run_one scripts/run_phase7_inception_v9.sh \
  checkpoints/fp_inception_resnet_v2_e35_bs96_t299_inception_v9_scale083_best.pth \
  inception_v9 \
  "FP inception v9 scale083 h256 pup120"

wait_for_gpu

run_one scripts/run_phase7_inception_v10.sh \
  checkpoints/fp_inception_resnet_v2_e35_bs96_t299_inception_v10_scale150_best.pth \
  inception_v10 \
  "FP inception v10 scale150 h256 pup120"

echo "Done. Blend strong runs only:"
echo "  python -m data.calibrate_submission submission/inception_v9_pup120.csv \\"
echo "    --blend submission/inception_v8_pup120.csv --weights 0.7,0.3 \\"
echo "    --output submission/blend_v9_v8.csv"
