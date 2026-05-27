# FP Report Outline (due 2026-05-31)

Fill this in for the course report. **Lead with best Kaggle private RMSE.**

## 1. Title and team

- Project: NOAA Steller Sea Lion Population Count
- Team members + contribution table (required)
- Repo: https://github.com/RayhanHaqi/visual_recognition-fp

## 2. Problem

- Input: aerial image (or tiles)
- Output: 5 non-negative counts (`adult_males`, `subadult_males`, `adult_females`, `juveniles`, `pups`)
- Metric: RMSE (lower is better)

## 3. Results table (required)

| Run | Method | Private RMSE | Public RMSE | Notes |
|-----|--------|--------------|-------------|-------|
| v2 | area labels, full Test | 28.09 | — | baseline train |
| v4 | v2 ckpt + Test×0.5 + shifts=5 + pup×1.2 | 25.54 | 26.55 | on-time backup |
| **v5** | lopuhin dots + balanced_dots | **17.41** | 17.93 | **best** |
| v6 | gaussian soft dots | TBD | TBD | Phase 3 |
| ensemble | v5+v4 blends | 17.61+ | — | worse than v5 alone |

Attach **Kaggle submissions screenshot** (Selected row).

## 4. Method (best pipeline: v5)

1. **Labels:** `data/coords-threeplusone-v0.4.csv` → `datasets/dot_labels.csv`
2. **Training:** ResNet50, `balanced_dots` tiles (dot-centered + background), 20 epochs, bs=256
3. **Inference:** `Test_scaled_0.5`, tile size 299, stride 299, `shifts=5`, sum tile preds → image
4. **Post:** `pups × 1.2` via `data.calibrate_submission`

## 5. Ablations / phases

| Phase | Idea | Outcome |
|-------|------|---------|
| 0 | scaled test + TTA + pup | v4 25.54 |
| 1 | lopuhin dot import + gate | stable labels |
| 2 | balanced_dots | **17.41** |
| 3 | gaussian_dots soft targets | TBD |
| 5 | ensemble v5+v4 | hurt (v4 too weak) |

## 6. Why local val misled

- Tile train RMSE ~1.5 vs image val RMSE ~35–45
- Kaggle image-level RMSE can improve when val worsens (v5 case)

## 7. Inference cost

- ~18k images, ~662 tiles/image, shifts=5 → ~4 h on 5090 (bs=512)
- Profile: preprocess ~62%, forward ~24% (`scripts/profile_inference.sh`)

## 8. Failure modes / future work

- Ensemble with weaker v4 hurt
- Phase 4 stronger backbone (optional)
- True heatmap head (lopuhin-style) not implemented

## 9. Repro commands (lab)

```bash
python scripts/import_dot_coords.py
SKIP_*=1 LABEL_MODE=balanced_dots bash scripts/run_phase1.sh resnet50 20 256 1e-4 1 299 0 balanced_dots_v5
python inference.py checkpoints/..._balanced_dots_v5_best.pth --test_subdir Test_scaled_0.5 --batch_size 512 --shifts 5
python -m data.calibrate_submission submission/balanced_dots_v5.csv --scale pups=1.2 --output submission/balanced_dots_v5_pup120.csv
```
