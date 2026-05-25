# Git LFS — dataset and training progress

Git LFS stores large files outside normal git history. This repo tracks:

| Path | Storage |
|---|---|
| `*.pth`, `*.pt` | LFS (checkpoints) |
| `*.7z` | LFS (archive, if committed) |
| `datasets/Train/**`, `TrainDotted/**`, `Test/**` | LFS (optional; **~96 GB**) |
| `datasets/train.csv`, `sample_submission.csv`, `MismatchedTrainImages.txt` | Regular git (small) |
| `log/*.csv`, `submission/*.csv` | Regular git |

## Important: do not push 96 GB of images unless you mean it

GitHub LFS billing applies to storage **and** bandwidth. The full image folders are ~96 GB.
For most workflows:

1. **Re-download images** on each machine with `setup.py` / `kaggle_curl_download.sh`.
2. **Push only progress**: checkpoints + CSV logs + small dataset metadata.

Only push `datasets/Train/` etc. if you accept LFS storage/bandwidth costs and long upload times.

## One-time setup (lab PC or local)

```bash
cd ~/Rayhan/selectedtopics/FP   # or your clone path
git lfs install
git pull
bash scripts/setup_git_lfs.sh
```

## Push training progress (recommended)

After a training run on w61:

```bash
cd ~/Rayhan/selectedtopics/FP
git lfs install
git pull

# Small label files (regular git)
git add datasets/train.csv datasets/sample_submission.csv datasets/MismatchedTrainImages.txt

# Progress artifacts
git add checkpoints/fp_resnet50_e30_bs128_t299_best.pth
git add log/fp_resnet50_e30_bs128_t299.csv
git add run_tracker.txt

# Optional: submission CSV when ready
# git add submission/fp_resnet50_e30_bs128_t299.csv

git status
git lfs status
git commit -m "Add FP training progress (epoch 1 best checkpoint + logs)."
git push origin main
```

Verify LFS before push:

```bash
git lfs ls-files
```

## Optional: push full dataset via LFS

**Warning:** ~96 GB upload; requires GitHub LFS data packs.

```bash
# Stop ignoring image folders (one-time edit) — or force-add:
git add -f datasets/Train datasets/TrainDotted datasets/Test

git lfs migrate info --include="datasets/**"
git status
git commit -m "Add NOAA sea lion images via Git LFS."
git push origin main
```

Prefer keeping images local and only pushing metadata + checkpoints.

## Pull on another machine

```bash
git lfs install
git pull
git lfs pull
```

Images still missing? Run `python setup.py --download` or extract the `.7z` locally.
