#!/bin/bash
# Pup×1.2 calibration + Kaggle submit for any phase run (v4/v5-style).
#
# Usage:
#   RUN_NAME=effnet_b3_v7 bash scripts/finish_phase_run.sh
#   RUN_NAME=inception_v8 SUBMIT_MSG="FP inception v8 pup120" bash scripts/finish_phase_run.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"
# shellcheck disable=SC1091
source "$(dirname "$0")/kaggle_env.sh"

RUN_NAME=${RUN_NAME:?Set RUN_NAME (submission base name, e.g. effnet_b3_v7)}
BEST_RMSE=${BEST_RMSE:-17.41298}
BEST_LABEL=${BEST_LABEL:-balanced_dots_v5_pup120}
RAW=submission/${RUN_NAME}.csv
PUP=submission/${RUN_NAME}_pup120.csv
SUBMIT_MSG=${SUBMIT_MSG:-"FP ${RUN_NAME} pup120"}

if [[ ! -f "$RAW" ]]; then
  echo "ERROR: missing $RAW"
  echo "Run inference first, e.g.:"
  echo "  CKPT=checkpoints/..._best.pth RUN_NAME=${RUN_NAME} AMP=1 bash scripts/run_infer_v5.sh"
  exit 1
fi

mkdir -p submission

echo "=================================================="
echo "FP finish — ${RUN_NAME} (pup scale + submit)"
echo "=================================================="
echo "Beat: ${BEST_LABEL} private RMSE ${BEST_RMSE}"
echo ""

python -m data.calibrate_submission "$RAW" \
  --scale pups=1.2 \
  --output "$PUP"

if [[ "${SKIP_SUBMIT:-0}" != "1" ]]; then
  bash scripts/submit.sh "$PUP" "$SUBMIT_MSG"
else
  echo "SKIP_SUBMIT=1; not submitting $PUP"
fi

echo ""
echo "Compare Kaggle private RMSE to ${BEST_RMSE}. Select best submission for report."
