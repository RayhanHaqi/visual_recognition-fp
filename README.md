# Final Project — NOAA Steller Sea Lion Population Count

NYCU Visual Recognition (2026 Spring), Topic 3: predict five sea-lion class counts per image; evaluated by **RMSE** on Kaggle.

**Workflow:** Develop and commit code on your machine; **train and submit on the lab PC** (e.g. RTX 5090). `datasets/` and `checkpoints/` are gitignored and stay on the lab machine only.

## Lab machine — first-time setup

```bash
git clone <your-repo-url> FP
cd FP
conda activate selectedtopics_env   # or: export FP_CONDA_ENV=visualrecognition
```

1. **Kaggle API** — join the [competition](https://www.kaggle.com/competitions/noaa-fisheries-steller-sea-lion-population-count), then add credentials **inside this repo** (recommended on shared lab PCs):

   ```bash
   mkdir -p .kaggle
   cp ~/Downloads/kaggle.json .kaggle/kaggle.json
   chmod 600 .kaggle/kaggle.json
   ```

   `.kaggle/` is gitignored — safe to clone the repo; do not commit `kaggle.json`.  
   Alternative: `~/.kaggle/kaggle.json` on a private machine only.

2. **Install + download + preprocess** (needs **≥110 GB** free disk):

```bash
python setup.py --install
python setup.py --download      # ~103 GB, may take hours
python setup.py --preprocess    # remove mismatched train images
python setup.py                 # verify layout
```

Or one flag at a time after clone:

```bash
python setup.py --install --download --preprocess
```

3. **Sanity tests:**

```bash
bash scripts/run_tests.sh
```

## Lab machine — Phase 1 (smoke + baseline + submit)

Full pipeline (recommended):

```bash
bash scripts/run_phase1.sh 0    # GPU id, default 0
```

Resume without repeating finished steps:

```bash
SKIP_DOWNLOAD=1 SKIP_PREPROCESS=1 SKIP_TESTS=1 SKIP_SMOKE=1 \
  bash scripts/run_phase1.sh 0
```

**Prediction contract:** model outputs per-tile counts; image-level prediction = **sum** of unique tile windows (`data/predict.py`, used by train val / `validate.py` / `inference.py`).

Or step by step:

```bash
# Optional smoke (~30–90 min)
python train.py --run_name smoke_v1 --epochs 5 --batch_size 16 --gpu 0 --use_tiles

# Baseline for first Kaggle submit (several hours)
python train.py --run_name baseline --epochs 30 --batch_size 16 --gpu 0 --use_tiles
python validate.py checkpoints/baseline_best.pth --gpu 0 --shifts 5
python inference.py checkpoints/baseline_best.pth --run_name baseline --gpu 0 --shifts 5
bash scripts/submit.sh submission/baseline.csv "FP baseline v1"
```

Shortcut train + infer only:

```bash
bash train.sh resnet50 30 16 1e-4 0 299
```

## What gets committed vs stays local

| Path | Git |
|------|-----|
| Code, `scripts/`, `tests/` | Yes |
| `datasets/` (~103 GB) | **No** — `setup.py --download` on lab |
| `checkpoints/`, `log/`, `submission/` | **No** — produced on lab |

## `setup.py` reference

```bash
python setup.py                  # install deps + check dataset
python setup.py --install        # pip install -r requirements.txt
python setup.py --download       # Kaggle download + extract → datasets/
python setup.py --preprocess     # drop MismatchedTrainImages
python setup.py --force-download # re-download if needed
```

Legacy wrapper: `bash scripts/download_data.sh` → `python setup.py --download`

## Layout

| Path | Purpose |
|------|---------|
| `data/` | PyTorch datasets, tiling, transforms |
| `model/` | timm backbone + 5-d count head |
| `utils/` | RMSE, splits, checkpoint I/O |
| `train.py` | Training |
| `inference.py` | Test predictions → CSV |
| `ensemble.py` | Average multiple submission CSVs |
| `scripts/run_phase1.sh` | End-to-end Phase 1 on lab GPU |

## Related work

- [lopuhin/kaggle-lions-2017](https://github.com/lopuhin/kaggle-lions-2017)
- [asanakoy/kaggle_sea_lions_counting](https://github.com/asanakoy/kaggle_sea_lions_counting)
