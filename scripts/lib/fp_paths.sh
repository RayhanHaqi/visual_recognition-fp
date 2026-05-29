#!/bin/bash
# Shared path helpers for FP shell scripts (source, do not execute).
# shellcheck shell=bash

# Match run_phase1.sh RUN_NAME: fp_${BACKBONE}_e${EPOCHS}_bs${BS}_t${TILE}_${SUFFIX}
fp_checkpoint_path() {
  local backbone=$1 epochs=$2 bs=$3 tile=$4 suffix=$5
  echo "checkpoints/fp_${backbone}_e${epochs}_bs${bs}_t${tile}_${suffix}_best.pth"
}

fp_require_checkpoint() {
  local ckpt=$1
  if [[ ! -f "$ckpt" ]]; then
    echo "ERROR: missing checkpoint: $ckpt"
    exit 1
  fi
}

fp_require_test_subdir() {
  local subdir=${1:-Test_scaled_0.5}
  if [[ ! -d "datasets/${subdir}" ]]; then
    echo "ERROR: missing datasets/${subdir}"
    echo "  python scripts/preprocess.py --downscale_test 0.5 --out_subdir ${subdir}"
    exit 1
  fi
  if ! compgen -G "datasets/${subdir}/*.jpg" >/dev/null; then
    echo "ERROR: datasets/${subdir} contains no .jpg test images"
    exit 1
  fi
}

fp_prepare_test_subdir() {
  local subdir=${1:-Test_scaled_0.5}
  if [[ -d "datasets/${subdir}" ]] && compgen -G "datasets/${subdir}/*.jpg" >/dev/null; then
    return 0
  fi
  if [[ "$subdir" =~ ^Test_scaled_([0-9]+(\.[0-9]+)?)$ ]]; then
    local scale=${BASH_REMATCH[1]}
    echo "Building datasets/${subdir} from Test at scale ${scale}..."
    python scripts/preprocess.py --downscale_test "$scale" --out_subdir "$subdir"
    fp_require_test_subdir "$subdir"
    return 0
  fi
  fp_require_test_subdir "$subdir"
}
