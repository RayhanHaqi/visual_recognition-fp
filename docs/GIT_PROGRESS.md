# What to commit to git

Keep large artifacts **local on the lab PC**. Only push small training outputs.

| Path | Commit? |
|---|---|
| Code, scripts, tests | Yes |
| `log/*.csv` | **Yes** — per-run metrics |
| `submission/*.csv` | **Yes** — Kaggle predictions |
| `datasets/` (~96 GB) | **No** — download with `setup.py` / `kaggle_curl_download.sh` |
| `checkpoints/*.pth` | **No** — re-train or copy manually (scp, Drive) |
| `*.7z` (e.g. `KaggleNOAASeaLions.7z`, `TrainSmall2.7z`) | **No** |
| `.kaggle/kaggle.json` | **No** |

## Push logs + submission after a run

```bash
cd ~/Rayhan/selectedtopics/FP
git pull

git add log/fp_resnet50_e30_bs128_t299.csv
git add submission/fp_resnet50_e30_bs128_t299.csv   # when ready

git status   # should NOT list datasets/ or checkpoints/
git commit -m "Add FP run logs and submission CSV."
git push origin main
```

## If you accidentally committed checkpoints or data

```bash
git rm --cached checkpoints/some_run_best.pth
git rm -r --cached datasets/   # if added by mistake
git commit -m "Stop tracking local checkpoints/datasets."
git push
```

Files stay on disk; git just stops tracking them.
