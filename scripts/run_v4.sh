#!/bin/bash
# v4 conservative leaderboard run: v2 checkpoint + downscaled Test inference.
set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"
# shellcheck disable=SC1091
source "$(dirname "$0")/kaggle_env.sh"

RUN_NAME=${RUN_NAME:-fp_resnet50_e30_bs128_t299_v4_test05}
CHECKPOINT=${CHECKPOINT:-checkpoints/fp_resnet50_e30_bs128_t299_v2_best.pth}
GPU=${GPU:-1}
SCALE=${SCALE:-0.5}
TEST_SUBDIR=${TEST_SUBDIR:-Test_scaled_${SCALE}}
SHIFTS=${SHIFTS:-5}
STRIDE=${STRIDE:-}
BATCH_SIZE=${BATCH_SIZE:-64}
SUBMIT_MSG=${SUBMIT_MSG:-}
if [[ -z "$SUBMIT_MSG" ]]; then
  SUBMIT_MSG="FP v4 v2 checkpoint test downscale ${SCALE}"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: missing checkpoint: $CHECKPOINT"
  echo "Copy or train the v2 best checkpoint on the lab PC first."
  exit 1
fi

if [[ ! -d "datasets/Test" ]]; then
  echo "ERROR: missing datasets/Test"
  echo "Run setup/download on the lab PC before v4 inference."
  exit 1
fi

needs_downscale=0
if [[ "${FORCE_DOWNSCALE:-0}" == "1" ]]; then
  needs_downscale=1
elif [[ ! -d "datasets/${TEST_SUBDIR}" ]] || [[ -z "$(ls -A "datasets/${TEST_SUBDIR}" 2>/dev/null)" ]]; then
  needs_downscale=1
fi

if [[ "$needs_downscale" == "1" ]]; then
  echo "--- Downscale Test x${SCALE} -> datasets/${TEST_SUBDIR} ---"
  python scripts/preprocess.py --downscale_test "$SCALE" --out_subdir "$TEST_SUBDIR"
else
  echo "--- Reusing datasets/${TEST_SUBDIR} (set FORCE_DOWNSCALE=1 to rebuild) ---"
fi

echo "--- Inference (${RUN_NAME}) ---"
PROFILE_ARGS=()
if [[ "${PROFILE_INFERENCE:-0}" == "1" ]]; then
  PROFILE_ARGS+=(--profile_inference)
fi
STRIDE_ARGS=()
if [[ -n "$STRIDE" ]]; then
  STRIDE_ARGS+=(--stride "$STRIDE")
fi

python inference.py "$CHECKPOINT" \
  --run_name "$RUN_NAME" \
  --gpu "$GPU" \
  --test_subdir "$TEST_SUBDIR" \
  --shifts "$SHIFTS" \
  --batch_size "$BATCH_SIZE" \
  "${STRIDE_ARGS[@]}" \
  "${PROFILE_ARGS[@]}"

CSV="submission/${RUN_NAME}.csv"

if [[ "${SKIP_SUBMIT:-0}" != "1" ]]; then
  echo "--- Kaggle submit ---"
  bash scripts/submit.sh "$CSV" "$SUBMIT_MSG"
else
  echo "SKIP_SUBMIT=1; not submitting $CSV"
fi

echo ""
echo "Done. Run: ${RUN_NAME}"
echo "  submission: ${CSV}"
echo ""
echo "Cheap pup-scale variant:"
echo "  python -m data.calibrate_submission ${CSV} --scale pups=1.2 --output submission/${RUN_NAME}_pup120.csv"
