# FP Report Outline (due 2026-05-31)

Fill this in for the course report. **Lead with best Kaggle private RMSE.**

**Living experiment record:** `docs/EXPERIMENT_LOG.md` — all trials, scores, and decision rationale (update after each submit; report pulls from here).

**Course checklist:** `docs/COURSE_REQUIREMENTS.md` — from `(114-2)FinalProject.pdf` + `Tips_FinalProj_Presentation.pdf` (deadlines, grading, mandatory screenshot/table/code link).

## 1. Title and team

- Project: NOAA Steller Sea Lion Population Count (course **Topic 3**, presentation **2026/06/02**)
- Report + code on **E3** by team leader: **2026/05/31 23:59** (report frozen after; slides may update)
- Team members + **contribution table** (required in report **and** slides — five tasks, % per member; see `COURSE_REQUIREMENTS.md`)
- Repo link (required): https://github.com/RayhanHaqi/visual_recognition-fp

## 2. Problem

- Input: aerial image (or tiles)
- Output: 5 non-negative counts (`adult_males`, `subadult_males`, `adult_females`, `juveniles`, `pups`)
- Metric: RMSE (lower is better)

## 3. Results table (required)

| Run | Method | Private RMSE | Public RMSE | Notes |
|-----|--------|--------------|-------------|-------|
| v2 | area labels, full Test | 28.09 | 29.10 | baseline train |
| v4 | v2 ckpt + Test×0.5 + pup×1.2 | 25.54 | 26.55 | on-time backup |
| v5 | ResNet-50 + balanced_dots + pup×1.2 | 17.41 | 17.93 | dot labels breakthrough |
| effnet_b3_v7 | EfficientNet-B3 + balanced_dots | 15.16 | 15.58 | Phase 4 |
| **inception_v8** | **Inception-ResNet v2 + balanced_dots** | **14.58** | **14.47** | **best** |
| v6 | gaussian soft dots | 27.40 | 28.40 | Phase 3 failed |
| ensemble | v5+v4 blends | 17.61+ | — | worse than v5 alone |

Attach **Kaggle submissions screenshot** (Selected row: `inception_v8_pup120`).

## 4. Method (best pipeline: inception_v8)

1. **Labels:** `data/coords-threeplusone-v0.4.csv` → `datasets/dot_labels.csv`
2. **Training:** Inception-ResNet v2, `balanced_dots` tiles, 20 epochs, bs=128
3. **Inference:** `Test_scaled_0.5`, tile 299, stride 299, `shifts=5`, sum tiles → image
4. **Post:** `pups × 1.2` via `data.calibrate_submission`

## 5. Ablations / phases

| Phase | Idea | Outcome |
|-------|------|---------|
| 0 | scaled test + TTA + pup | v4 25.54 |
| 1 | lopuhin dot import + gate | stable labels |
| 2 | balanced_dots (ResNet-50) | 17.41 |
| 3 | gaussian_dots | 27.40 (failed) |
| 4 | EfficientNet-B3 | 15.16 |
| 5 | ensemble v5+v4 | hurt |
| 6 | Inception-ResNet v2 | **14.58** |
| 7–7b | h256 head + scale aug | in progress |

## 6. Why local val misled

- Tile train RMSE ~0.2 vs image val RMSE ~35–45
- Kaggle can improve when val worsens (v5, v8)

## 7. Inference cost

- ~18k images, ~662 tiles/image, shifts=5 → ~4 h on 5090 (bs=512, AMP)
- Profile: preprocess ~62%, forward ~24%

## 8. Top-3 push (ongoing)

- Target: private RMSE < 13.05 (course **40 pt** band; golden historical ~15.88–10.86)
- `run_top3_quick_sweeps.sh`, `run_top3_train_queue.sh`
- MLP head + source-crop scale augmentation in code

## 9. Tips PDF — section goals (for full report marks)

| Section | Include |
|---------|---------|
| Introduction | Problem, importance, difficulty; how you advance the topic |
| Related work | **Groups** (tile CNN / UNet / ensembles); pros & cons each |
| Method | Pipeline overview figure; dot labels, Inception, infer, pup×1.2 |
| Results | Metric RMSE; **compare to** golden range & public solutions; **ablation table** (phases 0–7) |
| Conclusion | Best RMSE, main findings, lessons (val vs Kaggle, ensemble pitfalls) |

## 10. Presentation (06/02)

- **12 min** total (10 talk + 2 Q&A)
- Must show: Kaggle screenshot, contribution table, code link
- Slide filename: `Group[XX]_NOAA Fisheries Steller Sea Lion Population Count_[LEADER_ID].pptx` → course Google Drive

## 11. Repro commands (lab)

```bash
python scripts/import_dot_coords.py
LABEL_MODE=balanced_dots bash scripts/run_phase6_inception.sh
# infer + pup:
CKPT=checkpoints/fp_inception_resnet_v2_e20_bs128_t299_inception_v8_best.pth \
  RUN_NAME=inception_v8 AMP=1 bash scripts/run_infer_v5.sh
RUN_NAME=inception_v8 bash scripts/finish_phase_run.sh
```
