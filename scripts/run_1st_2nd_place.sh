#!/bin/bash
# 1st place (outrunner VGG-16) + 2nd place (Lopuhin UNet): train → infer → Kaggle submit.
#
#   tmux new -s fp_1st2nd
#   bash scripts/run_1st_2nd_place.sh
#
# Env:
#   PHASE=8|9|all (default all)   SKIP_PHASE8=1   SKIP_PHASE9=1   SKIP_WAIT=1
#   GPU=1   BEST_RMSE=14.44129
#
# Phase 8 only:  PHASE=8 bash scripts/run_1st_2nd_place.sh
# Phase 9 only:  PHASE=9 bash scripts/run_1st_2nd_place.sh
# Lopuhin UNet train (legacy torch 0.1): LOPUHIN_EXECUTE=1 PHASE=9 bash scripts/run_1st_2nd_place.sh
# Lopuhin CSV already built: LOPUHIN_CSV=path/to/raw.csv SKIP_LOPUHIN_TRAIN=1 PHASE=9 bash scripts/run_1st_2nd_place.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)
# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"
# shellcheck disable=SC1091
source "$(dirname "$0")/kaggle_env.sh"
# shellcheck disable=SC1091
source "$(dirname "$0")/lib/fp_paths.sh"

# --- config ---
PHASE=${PHASE:-all}
SKIP_PHASE8=${SKIP_PHASE8:-0}
SKIP_PHASE9=${SKIP_PHASE9:-0}
SKIP_WAIT=${SKIP_WAIT:-0}
ALLOW_MISSING_LOPUHIN=${ALLOW_MISSING_LOPUHIN:-0}
GPU=${GPU:-1}
BEST_RMSE=${BEST_RMSE:-14.44129}
INFER_BS=${INFER_BS:-256}
TEST_SUBDIR=${TEST_SUBDIR:-Test_scaled_0.5}

# Phase 8 — VGG / outrunner
P8_BACKBONE=${P8_BACKBONE:-vgg16}
P8_TILE=${P8_TILE:-300}
P8_EPOCHS=${P8_EPOCHS:-60}
P8_BS=${P8_BS:-64}
P8_LR=${P8_LR:-1e-4}
P8_RUN=${P8_RUN:-outrunner_vgg300}
P8_CKPT=$(fp_checkpoint_path "$P8_BACKBONE" "$P8_EPOCHS" "$P8_BS" "$P8_TILE" "$P8_RUN")
P8_MSG="FP outrunner vgg300 pup120"

# Phase 9 — Lopuhin
VENDOR=${VENDOR:-third_party/kaggle-lions-2017}
LOPUHIN_TAG=${LOPUHIN_TAG:-unet-stratified-scale-0.8-1.6-oversample0.2}
LOPUHIN_RUN_DIR=${LOPUHIN_RUN_DIR:-"$ROOT_DIR/$VENDOR/_runs/$LOPUHIN_TAG"}
LOPUHIN_RUN=${LOPUHIN_RUN:-lopuhin_unet}
LOPUHIN_MSG="FP lopuhin unet pup120"
LOPUHIN_EXECUTE=${LOPUHIN_EXECUTE:-0}
SKIP_LOPUHIN_TRAIN=${SKIP_LOPUHIN_TRAIN:-0}
LOPUHIN_CSV=${LOPUHIN_CSV:-}
LOPUHIN_SKIP_PUP_SCALE=${LOPUHIN_SKIP_PUP_SCALE:-0}

wait_gpu() {
  [[ "$SKIP_WAIT" == "1" ]] && return 0
  while pgrep -f 'train.py|inference.py' >/dev/null 2>&1; do
    pgrep -af 'train.py|inference.py' 2>/dev/null || true
    sleep 60
  done
}

# ========== Phase 8: train → infer → submit ==========
phase8() {
  echo "======== [1st] VGG-16 tile regression (outrunner) ========"
  export TILES_PER_IMAGE=${TILES_PER_IMAGE:-12}
  PHASE_TITLE="Phase 8 VGG t${P8_TILE}" \
    SKIP_INSTALL=1 SKIP_DOWNLOAD=1 SKIP_PREPROCESS=1 SKIP_SETUP=1 SKIP_TESTS=1 \
    SKIP_INFER=1 SKIP_SUBMIT=1 SKIP_VALIDATE=1 LABEL_MODE=balanced_dots \
    RUN_NAME= bash scripts/run_phase1.sh "$P8_BACKBONE" "$P8_EPOCHS" "$P8_BS" "$P8_LR" "$GPU" "$P8_TILE" 0 "$P8_RUN"
  [[ -f "$P8_CKPT" ]] || { echo "ERROR: missing $P8_CKPT"; exit 1; }

  PHASE_TITLE="infer $P8_RUN" CKPT="$P8_CKPT" RUN_NAME="$P8_RUN" GPU="$GPU" \
    TEST_SUBDIR="$TEST_SUBDIR" BATCH_SIZE="$INFER_BS" AMP=1 \
    bash scripts/run_infer_v5.sh

  RUN_NAME="$P8_RUN" SUBMIT_MSG="$P8_MSG" BEST_RMSE="$BEST_RMSE" \
    bash scripts/finish_phase_run.sh
  echo "Phase 8 done → submission/${P8_RUN}_pup120.csv"
}

# ========== Phase 9: vendor → (optional UNet) → submit ==========
setup_lopuhin() {
  mkdir -p third_party
  if [[ ! -d "$VENDOR/.git" ]]; then
    git clone --depth 1 https://github.com/lopuhin/kaggle-lions-2017.git "$VENDOR"
  fi
  [[ -d datasets ]] || { echo "ERROR: need datasets/ (setup.py --download)"; exit 1; }
  [[ -L "$VENDOR/data" || -d "$VENDOR/data" ]] || ln -sfn "$(pwd)/datasets" "$VENDOR/data"
  [[ -f data/coords-threeplusone-v0.4.csv ]] && \
    cp -n data/coords-threeplusone-v0.4.csv "$VENDOR/data/" 2>/dev/null || true
  mkdir -p "$LOPUHIN_RUN_DIR"
}

lopuhin_train_predict() {
  echo "LOPUHIN_EXECUTE=1 — vendor UNet (needs PyTorch 0.1 env in $VENDOR)"
  (cd "$VENDOR" && chmod +x ./unet.py 2>/dev/null || true
    ./unet.py "$LOPUHIN_RUN_DIR" --stratified --batch-size 32 \
      --min-scale 0.8 --max-scale 1.6 --n-epochs 13 --oversample 0.2
    ./unet.py "$LOPUHIN_RUN_DIR" --stratified --batch-size 32 \
      --min-scale 0.8 --max-scale 1.6 --n-epochs 13 --oversample 0.2 \
      --mode predict_all_valid
    python make_submission.py "$LOPUHIN_RUN_DIR" train --concat-features
    python make_submission.py "$LOPUHIN_RUN_DIR" predict --concat-features \
      --test-root "$LOPUHIN_RUN_DIR/test")
}

find_lopuhin_csv() {
  if [[ -n "$LOPUHIN_CSV" && -f "$LOPUHIN_CSV" ]]; then
    echo "$LOPUHIN_CSV"
    return 0
  fi
  local cand
  for cand in \
    "$LOPUHIN_RUN_DIR/${LOPUHIN_TAG}.csv" \
    "$LOPUHIN_RUN_DIR/$(basename "$LOPUHIN_RUN_DIR").csv" \
    "$VENDOR/submission.csv"; do
    if [[ -f "$cand" ]]; then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

phase9() {
  echo "======== [2nd] Lopuhin UNet + regressor ========"
  setup_lopuhin
  if [[ "$SKIP_LOPUHIN_TRAIN" != "1" && "$LOPUHIN_EXECUTE" == "1" ]]; then
    lopuhin_train_predict
  elif [[ "$SKIP_LOPUHIN_TRAIN" != "1" ]]; then
    echo "Skip UNet train (set LOPUHIN_EXECUTE=1 + legacy torch, or SKIP_LOPUHIN_TRAIN=1 + LOPUHIN_CSV)."
    echo "See docs/PHASE9_LOPUHIN_UNET.md"
  fi

  local raw
  if ! raw=$(find_lopuhin_csv); then
    echo "Phase 9: no submission CSV yet."
    echo "  Train: LOPUHIN_EXECUTE=1 PHASE=9 bash scripts/run_1st_2nd_place.sh"
    echo "  Or:    LOPUHIN_CSV=path/to.csv SKIP_LOPUHIN_TRAIN=1 PHASE=9 bash scripts/run_1st_2nd_place.sh"
    echo "  Docs:  docs/PHASE9_LOPUHIN_UNET.md"
    if [[ "$PHASE" == "9" || "$ALLOW_MISSING_LOPUHIN" != "1" ]]; then
      echo "ERROR: Phase 9 did not produce a CSV. Set SKIP_PHASE9=1 or ALLOW_MISSING_LOPUHIN=1 to continue intentionally."
      exit 1
    fi
    return 0
  fi
  mkdir -p submission
  cp "$raw" "submission/${LOPUHIN_RUN}.csv"
  if [[ "$LOPUHIN_SKIP_PUP_SCALE" == "1" ]]; then
    bash scripts/submit.sh "submission/${LOPUHIN_RUN}.csv" "$LOPUHIN_MSG raw"
    echo "Phase 9 done → submission/${LOPUHIN_RUN}.csv"
  else
    python -m data.calibrate_submission "submission/${LOPUHIN_RUN}.csv" \
      --scale pups=1.2 --output "submission/${LOPUHIN_RUN}_pup120.csv"
    bash scripts/submit.sh "submission/${LOPUHIN_RUN}_pup120.csv" "$LOPUHIN_MSG"
    echo "Phase 9 done → submission/${LOPUHIN_RUN}_pup120.csv"
  fi
}

# ========== main ==========
echo "=== 1st + 2nd place full pipeline | PHASE=$PHASE | beat RMSE $BEST_RMSE ===="
if [[ "$PHASE" == "8" || "$PHASE" == "all" ]] && [[ "$SKIP_PHASE8" != "1" ]]; then
  fp_prepare_test_subdir "$TEST_SUBDIR"
fi
wait_gpu

if [[ "$PHASE" == "8" || "$PHASE" == "all" ]] && [[ "$SKIP_PHASE8" != "1" ]]; then
  phase8
  wait_gpu
fi

if [[ "$PHASE" == "9" || "$PHASE" == "all" ]] && [[ "$SKIP_PHASE9" != "1" ]]; then
  phase9
fi

echo ""
echo "Check scores:"
echo "  kaggle competitions submissions -c noaa-fisheries-steller-sea-lion-population-count"
echo "Log trials in docs/EXPERIMENT_LOG.md"
