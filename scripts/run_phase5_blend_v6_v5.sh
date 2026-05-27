#!/bin/bash
# Phase 5b: blend v6 + v5 only (skip v4 — it hurts). Run after v6 pup120 is submitted.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=================================================="
echo "FP Phase 5b — v6 / v5 blend grid (no v4)"
echo "=================================================="

V6=${V6:-submission/gaussian_dots_v6_pup120.csv}
V5=${V5:-submission/balanced_dots_v5_pup120.csv}
OUT_DIR=${OUT_DIR:-submission}

for f in "$V6" "$V5"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f"
    exit 1
  fi
done

for w in 0.7 0.5 0.3; do
  v5w=$(python -c "print(round(1.0 - $w, 1))")
  out="${OUT_DIR}/blend_v6_${w}_v5_${v5w}.csv"
  python -m data.calibrate_submission "$V6" \
    --blend "$V5" \
    --weights "$w,$v5w" \
    --output "$out"
  echo "Wrote $out"
done

echo ""
echo "Submit 1-2 if v6 was close to v5, e.g.:"
echo "  bash scripts/submit.sh ${OUT_DIR}/blend_v6_0.5_v5_0.5.csv \"FP blend v6/v5 0.5\""
