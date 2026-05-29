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
# shellcheck disable=SC1091
source "$(dirname "$0")/lib/fp_paths.sh"

GPU=${GPU:-1}
BEST_RMSE=${BEST_RMSE:-14.44129}
INFER_BS=${INFER_BS:-256}
TEST_SUBDIR=${TEST_SUBDIR:-Test_scaled_0.5}
SKIP_WAIT=${SKIP_WAIT:-0}

# Must match run_phase7_inception_v9.sh / v10.sh defaults
V9_BACKBONE=${V9_BACKBONE:-inception_resnet_v2}
V9_EPOCHS=${V9_EPOCHS:-35}
V9_BS=${V9_BS:-96}
V9_TILE=${V9_TILE:-299}
V9_SUFFIX=${V9_SUFFIX:-inception_v9_scale083}
CKPT9=$(fp_checkpoint_path "$V9_BACKBONE" "$V9_EPOCHS" "$V9_BS" "$V9_TILE" "$V9_SUFFIX")

V10_BACKBONE=${V10_BACKBONE:-inception_resnet_v2}
V10_EPOCHS=${V10_EPOCHS:-35}
V10_BS=${V10_BS:-96}
V10_TILE=${V10_TILE:-299}
V10_SUFFIX=${V10_SUFFIX:-inception_v10_scale150}
CKPT10=$(fp_checkpoint_path "$V10_BACKBONE" "$V10_EPOCHS" "$V10_BS" "$V10_TILE" "$V10_SUFFIX")

wait_for_gpu() {
  [[ "$SKIP_WAIT" == "1" ]] && return 0
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
  local backbone=$5
  local epochs=$6
  local bs=$7
  local tile=$8
  local suffix=$9

  BACKBONE="$backbone" EPOCHS="$epochs" BS="$bs" TILE="$tile" SUFFIX="$suffix" GPU="$GPU" \
    RUN_NAME= bash "$train_script"
  fp_require_checkpoint "$ckpt"

  PHASE_TITLE="infer $run" \
    CKPT="$ckpt" RUN_NAME="$run" GPU="$GPU" \
    TEST_SUBDIR="$TEST_SUBDIR" BATCH_SIZE="$INFER_BS" AMP=1 \
    bash scripts/run_infer_v5.sh

  RUN_NAME="$run" SUBMIT_MSG="$msg" BEST_RMSE="$BEST_RMSE" \
    bash scripts/finish_phase_run.sh
}

echo "=== Top-3 train queue (target < ${BEST_RMSE}) ==="
echo "v9 ckpt:  $CKPT9"
echo "v10 ckpt: $CKPT10"
fp_prepare_test_subdir "$TEST_SUBDIR"
wait_for_gpu

run_one scripts/run_phase7_inception_v9.sh \
  "$CKPT9" \
  inception_v9 \
  "FP inception v9 scale083 h256 pup120" \
  "$V9_BACKBONE" "$V9_EPOCHS" "$V9_BS" "$V9_TILE" "$V9_SUFFIX"

wait_for_gpu

run_one scripts/run_phase7_inception_v10.sh \
  "$CKPT10" \
  inception_v10 \
  "FP inception v10 scale150 h256 pup120" \
  "$V10_BACKBONE" "$V10_EPOCHS" "$V10_BS" "$V10_TILE" "$V10_SUFFIX"

echo "Done. Blend strong runs only:"
echo "  python -m data.calibrate_submission submission/inception_v9_pup120.csv \\"
echo "    --blend submission/inception_v8_pup120.csv --weights 0.7,0.3 \\"
echo "    --output submission/blend_v9_v8.csv"
