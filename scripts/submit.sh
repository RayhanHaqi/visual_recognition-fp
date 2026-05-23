#!/bin/bash
# Submit to Kaggle after inference (requires ~/.kaggle/kaggle.json).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/kaggle_env.sh"

CSV=${1:-submission/baseline.csv}
MSG=${2:-FP baseline submission}

if [[ ! -f "$CSV" ]]; then
  echo "ERROR: missing $CSV — run inference.py first"
  exit 1
fi

kaggle competitions submit \
  -c noaa-fisheries-steller-sea-lion-population-count \
  -f "$CSV" \
  -m "$MSG"

echo "Submitted $CSV"
