#!/bin/bash
# End-to-end FP pipeline: setup → tests → train → validate → inference → Kaggle submit.
#
# Usage (same positional args as train.sh, optional 8th run suffix):
#   bash scripts/run_phase1.sh [backbone] [epochs] [batch] [lr] [gpu] [tile] [workers] [suffix]
#
# Examples:
#   bash scripts/run_phase1.sh resnet50 30 128 1e-4 1 299 0 v2
#   bash scripts/run_phase1.sh
#
# Skip steps (set env var to 1):
#   SKIP_INSTALL SKIP_DOWNLOAD SKIP_PREPROCESS SKIP_SETUP SKIP_TESTS
#   SKIP_TRAIN SKIP_VALIDATE SKIP_INFER SKIP_SUBMIT
#
# Restart training (kill stale train.py first): RESTART=1
# Override run name: RUN_NAME=my_run
#
set -euo pipefail

cd "$(dirname "$0")/.."

BACKBONE=${1:-resnet50}
EPOCHS=${2:-30}
BS=${3:-16}
LR=${4:-1e-4}
GPU=${5:-0}
TILE=${6:-299}
WORKERS=${7:-0}
RUN_SUFFIX=${8:-}

if [[ -n "${RUN_NAME:-}" ]]; then
  :
elif [[ -n "$RUN_SUFFIX" ]]; then
  RUN_NAME="fp_${BACKBONE}_e${EPOCHS}_bs${BS}_t${TILE}_${RUN_SUFFIX}"
else
  RUN_NAME="fp_${BACKBONE}_e${EPOCHS}_bs${BS}_t${TILE}"
fi

SHIFTS=${SHIFTS:-5}
SUBMIT_MSG=${SUBMIT_MSG:-"FP ${RUN_NAME}"}

LABEL_MODE=${LABEL_MODE:-area}
BUILD_DOT_CACHE=${BUILD_DOT_CACHE:-0}

TRAIN_EXTRA=()
if [[ "$LABEL_MODE" == "dots" || "$LABEL_MODE" == "balanced_dots" || "$LABEL_MODE" == "gaussian_dots" ]]; then
  TRAIN_EXTRA+=(--label_mode "$LABEL_MODE")
  if [[ "$BUILD_DOT_CACHE" == "1" ]]; then
    TRAIN_EXTRA+=(--build_dot_cache)
  fi
fi
if [[ -n "${HEAD_HIDDEN:-}" ]]; then
  TRAIN_EXTRA+=(--head_hidden "$HEAD_HIDDEN")
fi
if [[ -n "${DROPOUT:-}" ]]; then
  TRAIN_EXTRA+=(--dropout "$DROPOUT")
fi
if [[ -n "${SCALE_MIN:-}" && -n "${SCALE_MAX:-}" ]]; then
  TRAIN_EXTRA+=(--scale_min "$SCALE_MIN" --scale_max "$SCALE_MAX")
fi
if [[ -n "${TILES_PER_IMAGE:-}" ]]; then
  TRAIN_EXTRA+=(--tiles_per_image "$TILES_PER_IMAGE")
fi


# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"
# shellcheck disable=SC1091
source "$(dirname "$0")/kaggle_env.sh"
# shellcheck disable=SC1091
source "$(dirname "$0")/lib/fp_paths.sh"

PHASE_TITLE=${PHASE_TITLE:-Phase 1}
TEST_SUBDIR=${TEST_SUBDIR:-Test_scaled_0.5}
echo "=================================================="
echo "FP ${PHASE_TITLE} — ${RUN_NAME} | label=${LABEL_MODE} | GPU ${GPU} | env=${CONDA_DEFAULT_ENV:-system}"
echo "=================================================="

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  python setup.py --install
fi

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  python setup.py --download
fi

if [[ "${SKIP_PREPROCESS:-0}" != "1" ]]; then
  python setup.py --preprocess
fi

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  python setup.py
fi

if [[ "${SKIP_INFER:-0}" != "1" ]]; then
  fp_prepare_test_subdir "$TEST_SUBDIR"
fi

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  bash scripts/run_tests.sh
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if [[ "${RESTART:-0}" == "1" ]]; then
    echo "Stopping stale train.py jobs..."
    pkill -f 'train.py.*smoke_v1' 2>/dev/null || true
    pkill -f "train.py.*fp_${BACKBONE}" 2>/dev/null || true
    sleep 2
    pgrep -af 'train.py' || echo "(no train.py running)"
  fi

  echo "--- Train (${EPOCHS} epochs) ---"
  python train.py \
    --run_name "$RUN_NAME" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch_size "$BS" \
    --lr "$LR" \
    --tile_size "$TILE" \
    --gpu "$GPU" \
    --workers "$WORKERS" \
    --val_shifts 1 \
    --use_tiles \
    "${TRAIN_EXTRA[@]}"
fi

CKPT_PATH="checkpoints/${RUN_NAME}_best.pth"

if [[ "${SKIP_VALIDATE:-0}" != "1" ]]; then
  if [[ ! -f "$CKPT_PATH" ]]; then
    echo "ERROR: missing $CKPT_PATH (train first or set SKIP_TRAIN=0)"
    exit 1
  fi
  STRIDE_ARGS=()
  if [[ -n "${STRIDE:-}" ]]; then
    STRIDE_ARGS+=(--stride "$STRIDE")
  fi
  echo "--- Validate (tiled sum RMSE) ---"
  python validate.py "$CKPT_PATH" \
    --gpu "$GPU" \
    --shifts "$SHIFTS" \
    "${STRIDE_ARGS[@]}"
fi

if [[ "${SKIP_INFER:-0}" != "1" ]]; then
  if [[ ! -f "$CKPT_PATH" ]]; then
    echo "ERROR: missing $CKPT_PATH (train first or set SKIP_TRAIN=0)"
    exit 1
  fi
  fp_require_test_subdir "$TEST_SUBDIR"
  STRIDE_ARGS=()
  if [[ -n "${STRIDE:-}" ]]; then
    STRIDE_ARGS+=(--stride "$STRIDE")
  fi
  echo "--- Inference (${TEST_SUBDIR}) ---"
  python inference.py "$CKPT_PATH" \
    --run_name "$RUN_NAME" \
    --gpu "$GPU" \
    --test_subdir "$TEST_SUBDIR" \
    --shifts "$SHIFTS" \
    --batch_size "$BS" \
    "${STRIDE_ARGS[@]}"
fi

if [[ "${SKIP_SUBMIT:-0}" != "1" && "${SKIP_INFER:-0}" != "1" ]]; then
  echo "--- Kaggle submit ---"
  bash scripts/submit.sh "submission/${RUN_NAME}.csv" "$SUBMIT_MSG"
fi

echo ""
echo "Done. Run: ${RUN_NAME}"
echo "  checkpoint: checkpoints/${RUN_NAME}_best.pth"
echo "  log:        log/${RUN_NAME}.csv"
echo "  submission: submission/${RUN_NAME}.csv"
echo ""
echo "Resume with SKIP_*=1, e.g.:"
echo "  SKIP_INSTALL=1 SKIP_DOWNLOAD=1 SKIP_PREPROCESS=1 SKIP_SETUP=1 SKIP_TESTS=1 SKIP_TRAIN=1 \\"
echo "    bash scripts/run_phase1.sh ${BACKBONE} ${EPOCHS} ${BS} ${LR} ${GPU} ${TILE} ${WORKERS}${RUN_SUFFIX:+ ${RUN_SUFFIX}}"
echo ""
echo "Optional: bash scripts/push_progress.sh ${RUN_NAME}"
