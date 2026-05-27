#!/bin/bash
# After gaussian_dots_v6 inference: pup-scale, submit, remind score gate vs v5.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"

BEST_RMSE=${BEST_RMSE:-17.41298}
BEST_LABEL=${BEST_LABEL:-balanced_dots_v5_pup120}
RAW=submission/gaussian_dots_v6.csv
PUP=submission/gaussian_dots_v6_pup120.csv
SUBMIT_MSG=${SUBMIT_MSG:-"FP gaussian_dots v6 pup120"}

if [[ ! -f "$RAW" ]]; then
  echo "ERROR: missing $RAW"
  echo "Run inference first, e.g.:"
  echo "  PHASE_TITLE='Phase 3 infer' CKPT=checkpoints/fp_resnet50_e20_bs256_t299_gaussian_dots_v6_best.pth \\"
  echo "    RUN_NAME=gaussian_dots_v6 bash scripts/run_infer_v5.sh"
  exit 1
fi

echo "=================================================="
echo "FP Phase 3 — finish (pup scale + submit)"
echo "=================================================="
echo "Current best to beat: ${BEST_LABEL} private RMSE ${BEST_RMSE}"
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
echo "Check Kaggle submissions. If private RMSE < ${BEST_RMSE}, select v6 and optionally:"
echo "  bash scripts/run_phase5_blend_v6_v5.sh"
echo "Otherwise keep ${BEST_LABEL} as final."
