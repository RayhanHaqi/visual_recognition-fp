#!/bin/bash
# v3 on lab machine: dot labels, no dataset download (datasets/ must already exist).
set -euo pipefail

cd "$(dirname "$0")/.."

SKIP_INSTALL=1 SKIP_DOWNLOAD=1 SKIP_PREPROCESS=1 SKIP_SETUP=1 SKIP_TESTS=1 \
LABEL_MODE=dots BUILD_DOT_CACHE=1 \
  bash scripts/run_phase1.sh resnet50 30 128 1e-4 1 299 0 dots_v3
