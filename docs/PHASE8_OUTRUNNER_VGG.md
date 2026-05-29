# Phase 8 — 1st place (outrunner / VGG-16 tile regression)

**Status:** Prepared (runs on existing `train.py` + `inference.py`). Start when tile-CNN path plateaus (see `docs/EXPERIMENT_LOG.md` §1).

**References:** [yan-roo fork of outrunner kernel](https://github.com/yan-roo/Kaggle_Sea-Lions-Counting), Kaggle user `outrunner`.

---

## What 1st place did (target recipe)

| Item | outrunner (2017) | Our Phase 8 mapping |
|------|------------------|---------------------|
| Backbone | **VGG-16** ImageNet | `timm` `vgg16` via `--backbone vgg16` |
| Input | **300×300** patches | `--tile_size 300`, infer stride 300 |
| Loss | MSE on 5 counts | `train.py` MSE (default) |
| Labels | Dot counts per patch (+ background) | `balanced_dots` + Lopuhin `dot_labels.csv` |
| Optimizer | SGD 1e-4 → 1e-5, staged unfreeze | **v8a:** AdamW 1e-4 (fast). **v8b:** optional SGD script later |
| Epochs | ~60–220 (multi-stage) | **v8a:** 40–60 epochs; extend if LB improves |
| Test | Tiled predict + aggregate | `run_infer_v5.sh`, `Test_scaled_0.5`, `shifts=5` |
| Post | pups × 1.2 | `finish_phase_run.sh` |

Historical LB from notebook notes: private RMSE **~11.7** (2017). Course golden band: **15.88–10.86**.

---

## Why separate from Inception (Phase 6–7)

- Different **architecture** (shallow VGG vs Inception-ResNet).
- Different **tile size** (300 vs 299) — do not mix checkpoints.
- outrunner used **Lopuhin-style dots** in spirit; we already have stable CSV import.

---

## Lab commands

**Full train → infer → submit (recommended):**

```bash
tmux new -s fp_1st2nd
cd FP && conda activate selectedtopics_env
PHASE=8 bash scripts/run_1st_2nd_place.sh
```

Or both 1st + 2nd: `bash scripts/run_1st_2nd_place.sh` (see `docs/PHASE9_LOPUHIN_UNET.md` for Lopuhin UNet train).

**Optional stage-2 fine-tune** (all layers, lower LR) — only if v8a plateaus:

```bash
# TODO: add train.py --resume + lower LR; for now rerun with more epochs or manual ckpt continue
bash scripts/run_phase8_outrunner_vgg.sh   # increase EPOCHS=80
```

---

## Experiment IDs (log in `EXPERIMENT_LOG.md`)

| ID | Config | Notes |
|----|--------|-------|
| `outrunner_vgg300` | vgg16, t300, balanced_dots, 60ep, bs64 | Primary Phase 8 |
| `outrunner_vgg300_pup120` | + pup×1.2 submit | Production name after finish |
| `blend_vgg300_v8` | CSV blend with best Inception | Only if both strong |

---

## Risks

- VGG may **not** beat Inception-ResNet v2 on our dots (4th-place recipe already at 14.44 blend).
- **300px** infer must match train tile size.
- OOM: run `P8_BS=32 bash scripts/run_phase8_outrunner_vgg.sh`.

---

## Acceptance

- [ ] Checkpoint `..._outrunner_vgg300_best.pth` exists  
- [ ] Kaggle submit `outrunner_vgg300_pup120` scored  
- [ ] Row added to `EXPERIMENT_LOG.md` §2–3  
