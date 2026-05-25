#!/bin/bash
# Stop stale FP jobs and train only (no validate/infer/submit). Delegates to run_phase1.sh.
set -euo pipefail

cd "$(dirname "$0")/.."

BACKBONE=${1:-resnet50}
EPOCHS=${2:-30}
BS=${3:-128}
LR=${4:-1e-4}
GPU=${5:-0}
TILE=${6:-299}
WORKERS=${7:-0}
RUN_SUFFIX=${8:-v2}

RESTART=1 \
SKIP_INSTALL=1 SKIP_DOWNLOAD=1 SKIP_PREPROCESS=1 SKIP_SETUP=1 SKIP_TESTS=1 \
SKIP_VALIDATE=1 SKIP_INFER=1 SKIP_SUBMIT=1 \
  bash scripts/run_phase1.sh \
    "$BACKBONE" "$EPOCHS" "$BS" "$LR" "$GPU" "$TILE" "$WORKERS" "$RUN_SUFFIX"
