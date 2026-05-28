#!/bin/bash
# Phase 7: Inception-ResNet v2 + 256 hidden head + scale 0.83-1.25 (Asanakoy medium scale).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"

BACKBONE=${BACKBONE:-inception_resnet_v2}
BS=${BS:-96}
GPU=${GPU:-1}
EPOCHS=${EPOCHS:-35}

export HEAD_HIDDEN=256
export DROPOUT=0.5
export SCALE_MIN=0.83
export SCALE_MAX=1.25
export TILES_PER_IMAGE=16

PHASE_TITLE="Phase 7 — ${BACKBONE} scale083-125 h256" \
SKIP_INSTALL=1 SKIP_DOWNLOAD=1 SKIP_PREPROCESS=1 SKIP_SETUP=1 SKIP_TESTS=1 \
SKIP_INFER=1 SKIP_SUBMIT=1 SKIP_VALIDATE=1 \
LABEL_MODE=balanced_dots \
bash scripts/run_phase1.sh "$BACKBONE" "$EPOCHS" "$BS" 1e-4 "$GPU" 299 0 inception_v9_scale083
