# FP — NOAA Steller Sea Lion Population Count

## Where to run what

- **This repo (git):** code, scripts, tests only.
- **Lab PC (RTX 5090):** `python setup.py --download`, training, inference, Kaggle submit.
- **Never commit:** `datasets/`, `checkpoints/`, `log/`, `submission/` (see `.gitignore`).

## Competition
- URL: https://www.kaggle.com/competitions/noaa-fisheries-steller-sea-lion-population-count
- Task: Predict 5 non-negative counts per aerial image (regression, not bbox detection)
- Metric: **RMSE** (lower is better)
- Columns: `adult_males`, `adult_females`, `subadult_males`, `subadult_females`, `pups`

## Course grading (model performance, 50%)
- Golden medal (weak baseline): ~25 pts — historical golden ~15.88–10.86 RMSE
- Top 3: 40 pts
- Within 3% of best score: 50 pts (detection/counting band: 3%)

## Presentation
- Slot: **2026/06/02** (topic 3)
- Report due: **2026/05/31 23:59** (slides may update later; report cannot)
- Required in slides/report: Kaggle rank screenshot, team contribution table, code link

## Lab setup (clone → train)

```bash
export FP_CONDA_ENV=selectedtopics_env   # default in scripts/conda_env.sh
conda activate "$FP_CONDA_ENV"
cd FP
python setup.py --install --download --preprocess
bash scripts/run_tests.sh
bash scripts/run_phase1.sh 0
```

Resume: `SKIP_DOWNLOAD=1 SKIP_TRAIN=1 bash scripts/run_phase1.sh 0`

**Prediction:** per-tile counts → image total = **sum** of unique tiles (`data/predict.py`).

Requires `~/.kaggle/kaggle.json` (or env `KAGGLE_USERNAME` / `KAGGLE_KEY`) and **≥110 GB** free disk.

## Data layout (`datasets/`, ~103 GB, gitignored)
```
datasets/
├── Train/
├── TrainDotted/
├── Test/
├── train.csv
├── sample_submission.csv
├── MismatchedTrainImages.txt
└── coords-threeplusone-v0.4.csv   # optional
```

## Gotchas
1. Remove mismatched images before training: `python setup.py --preprocess`
2. Top solutions downscale **Test** to ~0.4–0.5× (`scripts/preprocess.py --downscale_test 0.5` + `inference.py --test_subdir Test_scaled`) — Phase 2
3. Multi-shift tile TTA (`--shifts 5`) improves RMSE
4. Pup +20% post-process (`--pup_scale 1.2`) — only after baseline LB
5. OOM on 5090 is unlikely at bs=16; fallback `--batch_size 8 --no_amp`

## Reference solutions (related work)
- https://github.com/lopuhin/kaggle-lions-2017 (UNet + regressor, 2nd place)
- https://github.com/asanakoy/kaggle_sea_lions_counting (tile Inception + ensemble)

## Repo conventions
- CLI via `argparse` only; dataset I/O via `setup.py` flags
- Metrics: `log/<run_name>.csv` (local only)
- Checkpoints: `checkpoints/<run_name>_best.pth` (local only)
- Submissions: `submission/<run_name>.csv` (local only)
