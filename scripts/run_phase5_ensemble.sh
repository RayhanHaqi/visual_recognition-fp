#!/bin/bash
# Phase 5: blend existing submissions (no training / inference).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=================================================="
echo "FP Phase 5 — ensemble (blend CSVs, no train/infer)"
echo "=================================================="

V5=${V5:-submission/balanced_dots_v5_pup120.csv}
V4=${V4:-submission/fp_resnet50_e30_bs128_t299_v4_test05_pup120.csv}
OUT_DIR=${OUT_DIR:-submission}

for w in 0.9 0.8 0.7 0.6 0.5; do
  v4w=$(python -c "print(round(1.0 - $w, 2))")
  out="${OUT_DIR}/ensemble_v5${w}_v4${v4w}.csv"
  python -m data.calibrate_submission "$V5" \
    --blend "$V4" \
    --weights "$w,$v4w" \
    --output "$out"
  echo "Wrote $out"
done

echo ""
echo "Submit promising blends, e.g.:"
echo "  bash scripts/submit.sh ${OUT_DIR}/ensemble_v50_v4.csv \"FP ensemble v5+v4 0.5\""
