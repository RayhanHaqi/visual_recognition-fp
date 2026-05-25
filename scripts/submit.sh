#!/bin/bash
# Submit to Kaggle after inference. Credentials: FP/.kaggle/kaggle.json (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/kaggle_env.sh"

CSV=${1:-submission/baseline.csv}
MSG=${2:-FP baseline submission}
SAMPLE=${SAMPLE:-datasets/sample_submission.csv}

if [[ ! -f "$CSV" ]]; then
  echo "ERROR: missing $CSV — run inference.py first"
  exit 1
fi

if [[ -f "$SAMPLE" ]]; then
  expected=$(($(wc -l < "$SAMPLE") - 1))
  actual=$(($(wc -l < "$CSV") - 1))
  if [[ "$expected" != "$actual" ]]; then
    echo "ERROR: $CSV has $actual rows but $SAMPLE requires $expected"
    echo "Restore official sample_submission.csv from the 7z archive, then run:"
    echo "  python -m data.expand_submission $CSV --template $SAMPLE --output ${CSV%.csv}_full.csv"
    exit 1
  fi
fi

kaggle competitions submit \
  -c noaa-fisheries-steller-sea-lion-population-count \
  -f "$CSV" \
  -m "$MSG"

echo "Submitted $CSV"
