# FP — NOAA Steller Sea Lion Population Count

## Where to run what

- **This repo (git):** code, scripts, tests only.
- **Lab PC (RTX 5090):** `python setup.py --download`, training, inference, Kaggle submit.
- **Never commit:** `datasets/`, `checkpoints/`, `log/`, `submission/` (see `.gitignore`).

## Competition
- URL: https://www.kaggle.com/competitions/noaa-fisheries-steller-sea-lion-population-count
- Task: Predict 5 non-negative counts per aerial image (regression, not bbox detection)
- Metric: **RMSE** (lower is better)
- Columns: `test_id`, `adult_males`, `subadult_males`, `adult_females`, `juveniles`, `pups`

## Course grading (model performance, 50%)
- Golden medal (weak baseline): ~25 pts — historical golden ~15.88–10.86 RMSE
- Top 3: 40 pts
- Within 3% of best score: 50 pts (detection/counting band: 3%)

## Presentation & course deliverables
- Slot: **2026/06/02** (topic 3 — NOAA sea lions)
- Report + code on **E3**: **2026/05/31 23:59** (late −20/day; **report cannot** be updated after; slides can)
- Presentation: **12 min** (10 + 2 Q&A)
- Required in **both** report and slides: Kaggle rank screenshot, team contribution table (5 tasks), code link
- Full checklist: `docs/COURSE_REQUIREMENTS.md` (from `(114-2)FinalProject.pdf`, `Tips_FinalProj_Presentation.pdf`)

## Lab setup (clone → train)

```bash
export FP_CONDA_ENV=selectedtopics_env   # default in scripts/conda_env.sh
conda activate "$FP_CONDA_ENV"
cd FP
python setup.py --install --download --preprocess
bash scripts/run_tests.sh
bash scripts/run_phase1.sh resnet50 30 128 1e-4 1 299 0 v2
```

Resume: `SKIP_INSTALL=1 SKIP_DOWNLOAD=1 SKIP_PREPROCESS=1 SKIP_SETUP=1 SKIP_TESTS=1 SKIP_TRAIN=1 bash scripts/run_phase1.sh resnet50 30 128 1e-4 1 299 0 v2`

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
└── dot_labels.csv                 # from import script (not hand-extracted)
```

**Dot labels (recommended):** use the committed community coords, not `python -m data.dots`:

```bash
python scripts/import_dot_coords.py
python scripts/analyze_dot_cache.py --data_path datasets --dot_cache datasets/dot_labels.csv --fail_on_gate
```

Source file in git: `data/coords-threeplusone-v0.4.csv` ([lopuhin/kaggle-lions-2017](https://github.com/lopuhin/kaggle-lions-2017)).

## Gotchas
1. Remove mismatched images before training: `python setup.py --preprocess`
2. Top solutions downscale **Test** to ~0.4–0.5× (`scripts/preprocess.py --downscale_test 0.5` + `inference.py --test_subdir Test_scaled`) — Phase 2
3. Multi-shift tile TTA (`--shifts 5`) improves RMSE
4. Pup +20% post-process (`--pup_scale 1.2`) — only after baseline LB
5. OOM on 5090 is unlikely at bs=16; fallback `--batch_size 8 --no_amp`

## Top-3 push (<13.05 private RMSE)

Current best: `inception_v8_pup120` (~14.58). Quick sweeps (no retrain):

```bash
bash scripts/run_top3_quick_sweeps.sh
```

Train Asanakoy-style Inception v9/v10 (256 hidden head + scale aug):

```bash
bash scripts/run_top3_train_queue.sh
# or one at a time: bash scripts/run_phase7_inception_v9.sh
```

Training knobs via env: `HEAD_HIDDEN=256 DROPOUT=0.5 SCALE_MIN=0.83 SCALE_MAX=1.25 TILES_PER_IMAGE=16`.

## Post-v5 backbones (lab, after GPU idle)

When `run_infer_v5.sh` / any `train.py` finishes, run EfficientNet-B3 then Inception-ResNet v2 (same `balanced_dots` as v5):

```bash
tmux new -s fp_p46
cd FP && conda activate "$FP_CONDA_ENV"
bash scripts/run_phases_4_and_6.sh
```

Single phase or resume: `PHASE=4 bash scripts/run_phases_4_and_6.sh` or `SKIP_PHASE4=1 bash scripts/run_phases_4_and_6.sh`. OOM: `BS=64` in `run_phase4.sh` / `run_phase6_inception.sh`, or `INFER_BS=128`.

Targets to beat: **14.441** private (`blend_v80.7_effnet0.3`); **14.577** (`inception_v8_pup120`).

**After tile-CNN ceiling (1st / 2nd place) — one script:**

```bash
bash scripts/run_1st_2nd_place.sh                # VGG train→infer→submit, then Lopuhin setup→submit
PHASE=8 bash scripts/run_1st_2nd_place.sh      # 1st place only
PHASE=9 LOPUHIN_CSV=path/to/raw.csv SKIP_LOPUHIN_TRAIN=1 bash scripts/run_1st_2nd_place.sh
```

Docs: `docs/PHASE8_OUTRUNNER_VGG.md`, `docs/PHASE9_LOPUHIN_UNET.md`

## Post-Phase-3 workflow (lab)

After `gaussian_dots_v6` train/infer:

```bash
bash scripts/post_phase3_workflow.sh   # or scripts/finish_phase3.sh
bash scripts/run_phase5_blend_v6_v5.sh # only if v6 is competitive with v5 (17.41)
```

Report skeleton: `docs/REPORT_OUTLINE.md`. Faster infer: `AMP=1` in `scripts/run_infer_v5.sh` or `python inference.py ... --amp`.

## Experiment documentation

- **Log every trial + decision:** `docs/EXPERIMENT_LOG.md` (update after each Kaggle submit).
- **Strategy:** current tile pipeline until RMSE plateaus, then 1st (VGG/outrunner) and 2nd (UNet/Lopuhin) per backlog in that file.
- **Course tasks / deadlines:** `docs/COURSE_REQUIREMENTS.md`
- Report skeleton: `docs/REPORT_OUTLINE.md`

## Reference solutions (related work)
- https://github.com/lopuhin/kaggle-lions-2017 (UNet + regressor, 2nd place)
- https://github.com/asanakoy/kaggle_sea_lions_counting (tile Inception + ensemble)

## Repo conventions
- CLI via `argparse` only; dataset I/O via `setup.py` flags
- Metrics: `log/<run_name>.csv` (local only)
- Checkpoints: `checkpoints/<run_name>_best.pth` (local only)
- Submissions: `submission/<run_name>.csv` (local only)
