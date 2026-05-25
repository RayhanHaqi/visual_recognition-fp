# Lab PC setup (e.g. w61) — full guide

One-page workflow for the NOAA sea lion FP on a shared lab machine. **Code via git; data and checkpoints stay local.**

## 0. Requirements

| Item | Detail |
|------|--------|
| Disk | **≥110 GB** free on the download volume |
| Conda env | `selectedtopics_env` (or set `FP_CONDA_ENV`) |
| Kaggle | Join [competition](https://www.kaggle.com/competitions/noaa-fisheries-steller-sea-lion-population-count) + accept rules |
| Tools | `p7zip-full` for extract (`sudo apt install p7zip-full`) |

**Do not use** the competition torrent / aria2 in 2026 — webseeds and trackers are dead; you get a full-size **invalid** file (`file` reports `data`, not `7-zip`).

---

## 1. Clone and credentials

```bash
git clone https://github.com/RayhanHaqi/visual_recognition-fp.git FP
cd FP
conda activate selectedtopics_env
```

Copy your token into the repo (USB / shared folder — **never commit**):

```bash
mkdir -p .kaggle
cp /path/to/kaggle.json .kaggle/kaggle.json
chmod 600 .kaggle/kaggle.json
```

**Always** before any `kaggle` command on the lab PC:

```bash
source scripts/kaggle_env.sh
```

Test auth (must **not** show `401`):

```bash
kaggle competitions files -c noaa-fisheries-steller-sea-lion-population-count
```

You should see `KaggleNOAASeaLions.7z` (~96 GB) in the list.

---

## 2. Download dataset (~96 GB, many hours)

### Recommended: streamed `curl` download (survives the Kaggle CLI RAM issue)

The normal `kaggle competitions download` command may buffer the 96 GB file in RAM before writing it. This script uses Kaggle's authenticated API endpoint with `curl --continue-at -`, so bytes stream directly to `KaggleNOAASeaLions.7z` and can resume.

```bash
cd ~/Rayhan/selectedtopics/FP
source scripts/kaggle_env.sh

rm -f Kaggle-NOAA-SeaLions Kaggle-NOAA-SeaLions.aria2   # bad aria2 leftovers

nohup bash scripts/kaggle_curl_download.sh KaggleNOAASeaLions.7z . \
  >> download_curl.log 2>&1 &

echo $!    # save PID
```

Monitor from another terminal:

```bash
watch -n 60 'ps -p <PID> -o etime,rss 2>/dev/null; ls -lh ~/Rayhan/selectedtopics/FP/KaggleNOAASeaLions.7z 2>&1; tail -3 ~/Rayhan/selectedtopics/FP/download_curl.log'
```

You should see `KaggleNOAASeaLions.7z` appear quickly and grow on disk. Target: `102894707185` bytes (~96 GB).

### Fallback: Kaggle CLI with `nohup`

```bash
cd ~/Rayhan/selectedtopics/FP
source scripts/kaggle_env.sh

rm -f Kaggle-NOAA-SeaLions Kaggle-NOAA-SeaLions.aria2   # bad aria2 leftovers

nohup kaggle competitions download \
  -c noaa-fisheries-steller-sea-lion-population-count \
  -f KaggleNOAASeaLions.7z \
  -p . \
  --force \
  >> download.log 2>&1 &

echo $!    # save PID, e.g. 208467
```

Monitor from another terminal (`Ctrl+C` only stops `tail`, not the download):

```bash
watch -n 60 'ps -p <PID> -o etime,rss 2>/dev/null; ls -lh ~/Rayhan/selectedtopics/FP/KaggleNOAASeaLions.7z 2>&1'
tail -20 ~/Rayhan/selectedtopics/FP/download.log
```

**Normal early behavior:** process runs, **no file** for 10–30+ minutes, RAM grows — then `KaggleNOAASeaLions.7z` appears and size increases.

**Target:** `102894707185` bytes (~96 GB). If no file appears and RSS keeps growing, stop this method and use the streamed `curl` script above.

### Alternative: `python setup.py --download`

Same Kaggle CLI under the hood, with a byte progress bar once the file exists.

### Alternative: tmux

```bash
tmux new -s sealion
source scripts/kaggle_env.sh
kaggle competitions download -c noaa-fisheries-steller-sea-lion-population-count -f KaggleNOAASeaLions.7z -p . --force
# Detach: Ctrl+b then d   (do not kill the tmux server)
tmux attach -t sealion
```

---

## 3. Verify before extract (required)

```bash
cd ~/Rayhan/selectedtopics/FP
ls -l KaggleNOAASeaLions.7z
file KaggleNOAASeaLions.7z
# MUST say: 7-zip archive data

7z t KaggleNOAASeaLions.7z -p"$(cat data_password.txt)" | tail -5
# MUST end with: Everything is Ok
```

If `file` says `data` or `7z t` fails → delete the file and re-download. **Do not extract.**

---

## 4. Extract and preprocess

```bash
PASS=$(cat data_password.txt)
mkdir -p datasets
7z x KaggleNOAASeaLions.7z -odatasets -p"$PASS"

# train.csv is NOT on Kaggle API — fetch corrected labels (not Train/train.csv in .7z)
python scripts/fetch_train_csv.py
source scripts/kaggle_env.sh
bash scripts/kaggle_curl_download.sh MismatchedTrainImages.txt datasets

head -1 datasets/train.csv
# expect: test_id,adult_males,subadult_males,adult_females,juveniles,pups

python setup.py --preprocess
python setup.py
```

Expected under `datasets/`: `Train/`, `TrainDotted/`, `Test/`, `train.csv` (Kaggle labels), etc.

Optional: remove archive after successful extract to save disk:

```bash
# rm KaggleNOAASeaLions.7z
```

---

## 5. Tests and Phase 1 training

```bash
bash scripts/run_tests.sh
bash scripts/run_phase1.sh resnet50 30 128 1e-4 1 299 0 v2
```

Or step by step:

```bash
python train.py --run_name fp_resnet50_e30_bs128_t299_v2 --epochs 30 --batch_size 128 --gpu 1 --tile_size 299 --val_shifts 1 --use_tiles
python -m data.fix_sample_submission --data_path datasets --limit 100
python validate.py checkpoints/fp_resnet50_e30_bs128_t299_v2_best.pth --gpu 1 --shifts 5 --stride 299
python inference.py checkpoints/fp_resnet50_e30_bs128_t299_v2_best.pth --run_name fp_resnet50_e30_bs128_t299_v2 --gpu 1 --shifts 5 --stride 299
bash scripts/submit.sh submission/fp_resnet50_e30_bs128_t299_v2.csv "FP resnet50 v2"
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| `401 Unauthorized` | Forgot `source scripts/kaggle_env.sh` or missing `.kaggle/kaggle.json` | Fix credentials; test `kaggle competitions files` |
| `^C User cancelled operation` | SIGINT (Ctrl+C, terminal stop, tmux killed) — not always literal Ctrl+C | Use `nohup`; don’t close tmux server |
| No `.7z` for 30+ min, RAM high | Kaggle buffering | Wait; if >60 min no file, check `download.log` and restart |
| Process gone, no file | tmux server died, OOM, or crash | `dmesg \| tail` (if allowed); restart `nohup` |
| `96G` file `Kaggle-NOAA-SeaLions`, `file` = `data` | Dead torrent / aria2 | `rm` it; download **`KaggleNOAASeaLions.7z`** via Kaggle API only |
| Inference ETA ~12 hours | Running model on every sample row | Use cached inference (latest code): only ~100 Test images are scored; decoy rows are zero-filled |
| Submission row count mismatch | Overwrote official sample with 100-row rebuild | Restore `datasets/sample_submission.csv` from 7z; expand with `python -m data.expand_submission` |
| Kaggle rejects columns | Wrong schema (`id`, `subadult_females`) | `python -m data.convert_submission submission/<run>.csv --output submission/<run>_kaggle.csv` |

---

## Command cheat sheet

```bash
source scripts/kaggle_env.sh                    # every new shell
kaggle competitions files -c <competition>      # auth + list (optional)
kaggle competitions download -c <competition> -f KaggleNOAASeaLions.7z -p .
file KaggleNOAASeaLions.7z && 7z t ...          # verify
7z x KaggleNOAASeaLions.7z -odatasets -p"$(cat data_password.txt)"
python setup.py --preprocess && python setup.py
bash scripts/run_phase1.sh resnet50 30 128 1e-4 1 299 0 v2
```
