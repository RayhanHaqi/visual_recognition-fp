#!/bin/bash
# No-code (existing ckpt) Kaggle sweeps: test scale, TTA, strong-model blends.
# Run on lab after: git pull && conda activate selectedtopics_env
#
#   tmux new -s fp_top3_sweep
#   bash scripts/run_top3_quick_sweeps.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"
# shellcheck disable=SC1091
source "$(dirname "$0")/kaggle_env.sh"

GPU=${GPU:-1}
CKPT=${CKPT:-checkpoints/fp_inception_resnet_v2_e20_bs128_t299_inception_v8_best.pth}
BEST_RMSE=${BEST_RMSE:-14.57729}
INFER_BS=${INFER_BS:-256}

echo "=== Top-3 quick sweeps (beat RMSE ${BEST_RMSE}) ==="

if [[ ! -d datasets/Test_scaled_0.4 ]]; then
  echo "Building Test_scaled_0.4..."
  python scripts/preprocess.py --downscale_test 0.4 --out_subdir Test_scaled_0.4
fi

echo "--- inception_v8 @ Test_scaled_0.4 ---"
PHASE_TITLE="top3 v8 test04" \
  CKPT="$CKPT" RUN_NAME=inception_v8_test04 GPU="$GPU" \
  TEST_SUBDIR=Test_scaled_0.4 BATCH_SIZE="$INFER_BS" AMP=1 \
  bash scripts/run_infer_v5.sh

RUN_NAME=inception_v8_test04 SUBMIT_MSG="FP inception v8 test04" \
  BEST_RMSE="$BEST_RMSE" bash scripts/finish_phase_run.sh

python -m data.calibrate_submission submission/inception_v8_test04.csv \
  --scale pups=1.2 --output submission/inception_v8_test04_pup120.csv
bash scripts/submit.sh submission/inception_v8_test04_pup120.csv "FP inception v8 test04 pup120"

if [[ -f submission/effnet_b3_v7_pup120.csv ]]; then
  echo "--- blend inception_v8_pup120 + effnet_b3_v7_pup120 ---"
  for w in 0.8 0.7 0.6; do
    v=$(python -c "print(round(1.0 - $w, 2))")
    out="submission/blend_v8${w}_effnet${v}.csv"
    python -m data.calibrate_submission submission/inception_v8_pup120.csv \
      --blend submission/effnet_b3_v7_pup120.csv \
      --weights "$w,$v" \
      --output "$out"
    bash scripts/submit.sh "$out" "FP blend v8 ${w} effnet ${v}"
  done
fi

echo "Done. Check: kaggle competitions submissions -c noaa-fisheries-steller-sea-lion-population-count"
