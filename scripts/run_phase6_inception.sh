#!/bin/bash
# Phase 6: Asanakoy-style backbone — Inception-ResNet v2 + balanced_dots (same pipeline as v5).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"

BACKBONE=${BACKBONE:-inception_resnet_v2}
BS=${BS:-128}
GPU=${GPU:-1}
EPOCHS=${EPOCHS:-20}
TILE=${TILE:-299}
SUFFIX=${SUFFIX:-inception_v8}

PHASE_TITLE="Phase 6 — ${BACKBONE} balanced_dots" \
SKIP_INSTALL=1 SKIP_DOWNLOAD=1 SKIP_PREPROCESS=1 SKIP_SETUP=1 SKIP_TESTS=1 \
SKIP_INFER=1 SKIP_SUBMIT=1 SKIP_VALIDATE=1 \
LABEL_MODE=balanced_dots \
RUN_NAME= bash scripts/run_phase1.sh "$BACKBONE" "$EPOCHS" "$BS" 1e-4 "$GPU" "$TILE" 0 "$SUFFIX"
