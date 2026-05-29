# Phase 9 — 2nd place (Lopuhin / UNet + secondary regressor)

**Status:** Prepared (vendor checkout + data symlinks). **Not** integrated into `train.py` yet — upstream code targets **PyTorch 0.1** (2017).

**Reference:** [lopuhin/kaggle-lions-2017](https://github.com/lopuhin/kaggle-lions-2017)

---

## What 2nd place did (two stages)

```text
Train images (scaled) ──► UNet ──► per-class density / segmentation maps
                                        │
                                        ▼
                         patch features (sums, blob_log thresholds)
                                        │
                                        ▼
                         XGBoost / ExtraTrees / Lasso per class
                                        │
                                        ▼
                              5 image-level counts
```

| Stage | Script (upstream) | GPU time (README) | Output |
|-------|-------------------|-------------------|--------|
| 1 — UNet | `unet.py _runs/...` | ~20 h, ~8 GB VRAM | `best-model.pt`, prediction pickles |
| 2 — Regressor | `make_submission.py` | CPU-heavy | Kaggle CSV |

We **already use** their `coords-threeplusone-v0.4.csv` for tile training. Phase 9 adds the **full density + regressor** stack.

---

## Integration options

| Option | Effort | When |
|--------|--------|------|
| **A — Vendor run** | Low setup, high env pain | Clone to `third_party/`, separate old PyTorch venv (often impractical on RTX 5090) |
| **B — Modern port** | High | Reimplement `unet_models.py` in PyTorch 2 + `train_unet.py` in this repo (recommended long-term) |
| **C — Hybrid** | Medium | Run UNet training on CPU/cloud with archived env; port only `make_submission` feature extraction |

**Prepared now:** Option A scaffolding (`scripts/setup_lopuhin_vendor.sh`, `scripts/run_phase9_lopuhin_unet.sh`). Option B tracked as backlog in `docs/EXPERIMENT_LOG.md` §5.

---

## One command (submit path)

```bash
# Full pipeline with Phase 8 + Phase 9 (Lopuhin submit needs CSV — see below):
bash scripts/run_1st_2nd_place.sh

# Phase 9 only, CSV already from vendor training:
PHASE=9 SKIP_LOPUHIN_TRAIN=1 LOPUHIN_CSV=third_party/kaggle-lions-2017/_runs/.../....csv \
  bash scripts/run_1st_2nd_place.sh

# Try vendor UNet train (legacy PyTorch 0.1 only):
LOPUHIN_EXECUTE=1 PHASE=9 bash scripts/run_1st_2nd_place.sh
```

`LOPUHIN_CSV` should be the raw, unscaled CSV. If the CSV is already pup-scaled, add
`LOPUHIN_SKIP_PUP_SCALE=1` to avoid applying pup scaling twice.

**Vendor setup only:** `bash scripts/setup_lopuhin_vendor.sh`

Creates:

```text
third_party/kaggle-lions-2017/   # git clone (gitignored)
  data -> symlink to ../../datasets + coords CSV
  _runs/                           # training outputs (gitignored)
```

---

## Upstream train / predict (manual, after setup)

From vendor README (params must match train vs predict):

```bash
cd third_party/kaggle-lions-2017
# Requires Python 3.5 + torch 0.1.12 — see README.rst; may need legacy Docker

./unet.py _runs/unet-stratified-scale-0.8-1.6-oversample0.2 \
  --stratified --batch-size 32 \
  --min-scale 0.8 --max-scale 1.6 \
  --n-epochs 13 --oversample 0.2

./unet.py _runs/unet-stratified-scale-0.8-1.6-oversample0.2 \
  ...same args... --mode predict_all_valid

python make_submission.py ...   # see vendor repo for full command chain
```

Copy resulting CSV to `submission/lopuhin_unet_pup120.csv` and apply pup scale if needed:

```bash
cd ../..
python -m data.calibrate_submission submission/lopuhin_unet.csv \
  --scale pups=1.2 --output submission/lopuhin_unet_pup120.csv
bash scripts/submit.sh submission/lopuhin_unet_pup120.csv "FP lopuhin unet pup120"
```

---

## Modern port plan (Option B — future `train_unet.py`)

1. `model/unet_lopuhin.py` — UNet + head from `unet_models.py` (rewrite for torch 2).  
2. `data/unet_dataset.py` — stratified patches, Gaussian/dot masks (`SegmentationDataset` logic).  
3. `train_unet.py` — CLI matching repo conventions (`argparse`, `log/`, `checkpoints/`).  
4. `predict_unet.py` — write per-image `.npy` preds like vendor.  
5. `regressor/make_submission_features.py` — port feature extraction from `make_submission.py` (sklearn + xgboost).  
6. Wire `finish_phase_run.sh` for final CSV.

Dependencies to add when porting: `scikit-image`, `xgboost`, `shapely` (see vendor `requirements.txt`).

---

## Experiment IDs

| ID | Description |
|----|-------------|
| `lopuhin_unet` | Vendor or ported UNet + regressor submit |
| `lopuhin_unet_pup120` | + pup×1.2 |
| `blend_lopuhin_v8` | Blend with best tile CNN if competitive |

---

## Acceptance

- [ ] `bash scripts/setup_lopuhin_vendor.sh` succeeds  
- [ ] UNet checkpoint or vendor `_runs/.../best-model.pt` exists  
- [ ] Submission CSV on Kaggle + logged in `EXPERIMENT_LOG.md`  
