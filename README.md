# Final Project — NOAA Steller Sea Lion Population Count

NYCU Visual Recognition (2026 Spring), Topic 3: predict five sea-lion class counts per image; evaluated by **RMSE** on Kaggle.

**Workflow:** Code on git; **train and download on the lab PC** (e.g. w61). `datasets/` and `checkpoints/` are gitignored.

**Full lab walkthrough:** [docs/LAB_SETUP.md](docs/LAB_SETUP.md)

---

## Lab PC — quick start

```bash
git clone https://github.com/RayhanHaqi/visual_recognition-fp.git FP
cd FP
conda activate selectedtopics_env

mkdir -p .kaggle
cp /path/to/kaggle.json .kaggle/kaggle.json    # copy manually; never commit
chmod 600 .kaggle/kaggle.json

source scripts/kaggle_env.sh
kaggle competitions files -c noaa-fisheries-steller-sea-lion-population-count   # must not 401

sudo apt install p7zip-full    # once
```

### Download (~96 GB, use streamed `curl` + `nohup`)

```bash
cd FP
source scripts/kaggle_env.sh

nohup bash scripts/kaggle_curl_download.sh KaggleNOAASeaLions.7z . \
  >> download_curl.log 2>&1 &
echo $!   # save PID

# other terminal:
watch -n 60 'ps -p <PID> -o etime,rss; ls -lh KaggleNOAASeaLions.7z 2>&1; tail -3 download_curl.log'
```

The `curl` script uses Kaggle auth from `.kaggle/kaggle.json`, streams directly to disk, and resumes partial downloads. **Do not use** `Kaggle-NOAA-SeaLions.torrent` / aria2 — dead in 2026; produces a useless ~96 GB `data` file.

### Verify → extract → preprocess

```bash
file KaggleNOAASeaLions.7z          # must: 7-zip archive
7z t KaggleNOAASeaLions.7z -p"$(cat data_password.txt)"

mkdir -p datasets
7z x KaggleNOAASeaLions.7z -odatasets -p"$(cat data_password.txt)"

# train.csv is a separate Kaggle file (not Train/train.csv inside the .7z)
bash scripts/kaggle_curl_download.sh train.csv datasets
bash scripts/kaggle_curl_download.sh MismatchedTrainImages.txt datasets

python setup.py --preprocess
python setup.py
```

### Train and submit

```bash
bash scripts/run_tests.sh
bash scripts/run_phase1.sh 0
```

---

## `setup.py` (optional wrapper)

```bash
python setup.py --install
python setup.py --download       # Kaggle CLI + progress bar + extract
python setup.py --small-download # TrainSmall2.7z smoke (~99 MB)
python setup.py --preprocess
python setup.py --force-download
```

---

## Phase 1 (manual steps)

```bash
python train.py --run_name baseline --epochs 30 --batch_size 16 --gpu 0 --use_tiles
python validate.py checkpoints/baseline_best.pth --gpu 0 --shifts 5
python inference.py checkpoints/baseline_best.pth --run_name baseline --gpu 0 --shifts 5
bash scripts/submit.sh submission/baseline.csv "FP baseline v1"
```

Resume pipeline:

```bash
SKIP_DOWNLOAD=1 SKIP_PREPROCESS=1 SKIP_TESTS=1 SKIP_SMOKE=1 \
  bash scripts/run_phase1.sh 0
```

**Prediction contract:** per-tile counts; image prediction = **sum** of unique tile windows (`data/predict.py`).

---

## Git vs local

| Path | Git |
|------|-----|
| Code, `scripts/`, `tests/` | Yes |
| `.kaggle/kaggle.json` | **No** |
| `datasets/` (~100+ GB after extract) | **No** |
| `checkpoints/`, `log/`, `submission/` | **No** |

---

## Layout

| Path | Purpose |
|------|---------|
| `data/` | Datasets, tiling, transforms, shared inference |
| `model/` | timm backbone + 5-d head |
| `train.py` / `inference.py` | Train and Kaggle CSV |
| `scripts/kaggle_env.sh` | `KAGGLE_CONFIG_DIR=FP/.kaggle` |
| `scripts/run_phase1.sh` | End-to-end Phase 1 |

## Related work

- [lopuhin/kaggle-lions-2017](https://github.com/lopuhin/kaggle-lions-2017)
- [asanakoy/kaggle_sea_lions_counting](https://github.com/asanakoy/kaggle_sea_lions_counting)
