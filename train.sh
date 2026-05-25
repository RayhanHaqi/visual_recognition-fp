#!/bin/bash
# Train + inference (no Kaggle submit). Delegates to run_phase1.sh.
set -euo pipefail

cd "$(dirname "$0")"

SKIP_INSTALL=1 SKIP_DOWNLOAD=1 SKIP_PREPROCESS=1 SKIP_SETUP=1 SKIP_TESTS=1 \
SKIP_VALIDATE=1 SKIP_SUBMIT=1 \
  bash scripts/run_phase1.sh "$@"
