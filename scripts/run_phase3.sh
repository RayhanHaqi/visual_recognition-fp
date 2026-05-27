#!/bin/bash
# Phase 3: gaussian soft-dot tile targets (same inference path as v5).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"

SKIP_INSTALL=1 SKIP_DOWNLOAD=1 SKIP_PREPROCESS=1 SKIP_SETUP=1 SKIP_TESTS=1 \
SKIP_INFER=1 SKIP_SUBMIT=1 SKIP_VALIDATE=1 \
LABEL_MODE=gaussian_dots \
bash scripts/run_phase1.sh resnet50 20 256 1e-4 1 299 0 gaussian_dots_v6
