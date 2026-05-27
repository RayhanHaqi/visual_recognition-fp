#!/bin/bash
# Post-Phase-3 checklist: finish v6 submit, optional v6/v5 blends, report reminder.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Post-Phase-3 workflow"
echo "====================="
echo "1) Ensure gaussian_dots_v6 inference completed (submission/gaussian_dots_v6.csv)"
echo "2) finish_phase3.sh — pup scale + Kaggle submit"
echo "3) If v6 private RMSE within ~0.5 of 17.41: run_phase5_blend_v6_v5.sh"
echo "4) Draft report from docs/REPORT_OUTLINE.md"
echo "5) Phase 4 only if report draft exists and v6 did not beat v5"
echo ""

if [[ -f submission/gaussian_dots_v6.csv ]]; then
  echo "Found submission/gaussian_dots_v6.csv — running finish_phase3.sh"
  bash scripts/finish_phase3.sh
else
  echo "No gaussian_dots_v6.csv yet. Run infer in tmux first:"
  echo "  tmux new -s fp_v6_infer"
  echo "  PHASE_TITLE='Phase 3 infer' \\"
  echo "    CKPT=checkpoints/fp_resnet50_e20_bs256_t299_gaussian_dots_v6_best.pth \\"
  echo "    RUN_NAME=gaussian_dots_v6 bash scripts/run_infer_v5.sh"
fi
