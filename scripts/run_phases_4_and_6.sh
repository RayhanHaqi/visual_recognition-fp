#!/bin/bash
# After GPU is free: train → infer → pup×1.2 submit for Phase 4 (EfficientNet-B3) and Phase 6 (Inception-ResNet v2).
#
# Lab (tmux recommended):
#   tmux new -s fp_p46
#   cd ~/Rayhan/selectedtopics/FP && conda activate selectedtopics_env
#   bash scripts/run_phases_4_and_6.sh
#
# Resume mid-pipeline:
#   SKIP_WAIT=1 SKIP_PHASE4=1 bash scripts/run_phases_4_and_6.sh   # only Phase 6
#   SKIP_WAIT=1 PHASE=4 bash scripts/run_phases_4_and_6.sh           # only Phase 4
#
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"
# shellcheck disable=SC1091
source "$(dirname "$0")/lib/fp_paths.sh"

GPU=${GPU:-1}
BEST_RMSE=${BEST_RMSE:-17.41298}
BEST_LABEL=${BEST_LABEL:-balanced_dots_v5_pup120}
INFER_BS=${INFER_BS:-256}
TEST_SUBDIR=${TEST_SUBDIR:-Test_scaled_0.5}
WAIT_FOR_GPU=${WAIT_FOR_GPU:-1}
SKIP_PHASE4=${SKIP_PHASE4:-0}
SKIP_PHASE6=${SKIP_PHASE6:-0}
PHASE=${PHASE:-all}

P4_BACKBONE=${P4_BACKBONE:-tf_efficientnet_b3_ns}
P4_EPOCHS=${P4_EPOCHS:-20}
P4_BS=${P4_BS:-${BS:-128}}
P4_TILE=${P4_TILE:-299}
P4_SUFFIX=${P4_SUFFIX:-effnet_b3_balanced_v7}
CKPT4=$(fp_checkpoint_path "$P4_BACKBONE" "$P4_EPOCHS" "$P4_BS" "$P4_TILE" "$P4_SUFFIX")

P6_BACKBONE=${P6_BACKBONE:-inception_resnet_v2}
P6_EPOCHS=${P6_EPOCHS:-20}
P6_BS=${P6_BS:-${BS:-128}}
P6_TILE=${P6_TILE:-299}
P6_SUFFIX=${P6_SUFFIX:-inception_v8}
CKPT6=$(fp_checkpoint_path "$P6_BACKBONE" "$P6_EPOCHS" "$P6_BS" "$P6_TILE" "$P6_SUFFIX")
RUN4=effnet_b3_v7
RUN6=inception_v8

wait_for_gpu() {
  echo "Waiting until no train.py / inference.py processes..."
  while pgrep -f 'train.py|inference.py' >/dev/null 2>&1; do
    pgrep -af 'train.py|inference.py' 2>/dev/null || true
    sleep 60
  done
  echo "GPU appears idle."
}

run_phase4() {
  echo ""
  echo "======== Phase 4: EfficientNet-B3 + balanced_dots ========"
  BACKBONE="$P4_BACKBONE" EPOCHS="$P4_EPOCHS" BS="$P4_BS" TILE="$P4_TILE" \
    SUFFIX="$P4_SUFFIX" GPU="$GPU" RUN_NAME= bash scripts/run_phase4.sh

  if [[ ! -f "$CKPT4" ]]; then
    echo "ERROR: missing $CKPT4 after training"
    exit 1
  fi

  PHASE_TITLE="Phase 4 infer" \
    CKPT="$CKPT4" RUN_NAME="$RUN4" GPU="$GPU" \
    TEST_SUBDIR="$TEST_SUBDIR" BATCH_SIZE="$INFER_BS" AMP=1 \
    bash scripts/run_infer_v5.sh

  RUN_NAME="$RUN4" SUBMIT_MSG="FP effnet b3 v7 pup120" \
    BEST_RMSE="$BEST_RMSE" BEST_LABEL="$BEST_LABEL" \
    bash scripts/finish_phase_run.sh
}

run_phase6() {
  echo ""
  echo "======== Phase 6: Inception-ResNet v2 + balanced_dots ========"
  BACKBONE="$P6_BACKBONE" EPOCHS="$P6_EPOCHS" BS="$P6_BS" TILE="$P6_TILE" \
    SUFFIX="$P6_SUFFIX" GPU="$GPU" RUN_NAME= bash scripts/run_phase6_inception.sh

  if [[ ! -f "$CKPT6" ]]; then
    echo "ERROR: missing $CKPT6 after training"
    exit 1
  fi

  PHASE_TITLE="Phase 6 infer" \
    CKPT="$CKPT6" RUN_NAME="$RUN6" GPU="$GPU" \
    TEST_SUBDIR="$TEST_SUBDIR" BATCH_SIZE="$INFER_BS" AMP=1 \
    bash scripts/run_infer_v5.sh

  RUN_NAME="$RUN6" SUBMIT_MSG="FP inception v8 pup120" \
    BEST_RMSE="$BEST_RMSE" BEST_LABEL="$BEST_LABEL" \
    bash scripts/finish_phase_run.sh
}

echo "=================================================="
echo "FP Phases 4 + 6 — backbones after v5 (beat RMSE ${BEST_RMSE})"
echo "=================================================="
fp_prepare_test_subdir "$TEST_SUBDIR"

if [[ "$WAIT_FOR_GPU" == "1" && "${SKIP_WAIT:-0}" != "1" ]]; then
  wait_for_gpu
fi

if [[ "$PHASE" == "4" || "$PHASE" == "all" ]]; then
  if [[ "$SKIP_PHASE4" != "1" ]]; then
    run_phase4
  else
    echo "SKIP_PHASE4=1"
  fi
fi

if [[ "$PHASE" == "6" || "$PHASE" == "all" ]]; then
  if [[ "$SKIP_PHASE6" != "1" ]]; then
    run_phase6
  else
    echo "SKIP_PHASE6=1"
  fi
fi

echo ""
echo "Done. Check Kaggle for:"
echo "  submission/${RUN4}_pup120.csv"
echo "  submission/${RUN6}_pup120.csv"
echo "Keep ${BEST_LABEL} unless a new private RMSE beats ${BEST_RMSE}."
