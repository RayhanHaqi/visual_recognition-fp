#!/bin/bash
# Smoke profile: timings on a few test images without a full 14h run.
set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"

CHECKPOINT=${CHECKPOINT:-checkpoints/fp_resnet50_e30_bs128_t299_v2_best.pth}
GPU=${GPU:-1}
TEST_SUBDIR=${TEST_SUBDIR:-Test_scaled_0.5}
SHIFTS=${SHIFTS:-5}
STRIDE=${STRIDE:-299}
BATCH_SIZE=${BATCH_SIZE:-512}
MAX_IMAGES=${MAX_IMAGES:-20}
RUN_NAME=${RUN_NAME:-infer_profile_smoke}

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: missing checkpoint: $CHECKPOINT"
  exit 1
fi

python inference.py "$CHECKPOINT" \
  --run_name "$RUN_NAME" \
  --gpu "$GPU" \
  --test_subdir "$TEST_SUBDIR" \
  --shifts "$SHIFTS" \
  --stride "$STRIDE" \
  --batch_size "$BATCH_SIZE" \
  --max_images "$MAX_IMAGES" \
  --profile_inference \
  --output "submission/${RUN_NAME}.csv"

echo ""
echo "Profile CSV: log/${RUN_NAME}_infer_profile.csv"
echo "Extrapolate: hours ≈ (18636 / images_per_sec) / 3600"
