#!/bin/bash
# Commit and push training logs + submission CSVs only (no checkpoints/datasets/7z).
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/push_progress.sh <run_name> [git commit message]"
  echo "Example: bash scripts/push_progress.sh fp_resnet50_e30_bs128_t299_v2"
  exit 1
fi

RUN_NAME="$1"
MSG="${2:-Add FP logs and submission for ${RUN_NAME}.}"

LOG_CSV="log/${RUN_NAME}.csv"
SUB_CSV="submission/${RUN_NAME}.csv"

git pull

to_add=()
[[ -f "$LOG_CSV" ]] && to_add+=("$LOG_CSV")
[[ -f "$SUB_CSV" ]] && to_add+=("$SUB_CSV")

if [[ ${#to_add[@]} -eq 0 ]]; then
  echo "ERROR: nothing to add. Expected at least: $LOG_CSV"
  exit 1
fi

git add "${to_add[@]}"
git status

if git diff --cached --quiet; then
  echo "Nothing new to commit."
  exit 0
fi

git commit -m "$MSG"
git push origin main
echo "Pushed: ${to_add[*]}"
