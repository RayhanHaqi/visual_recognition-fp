# Experiment log — NOAA sea lion count (FP)

**Policy (2026-05):** Push the **current tile-regression pipeline** (Asanakoy-style Inception + Lopuhin dots) until scores **plateau**. Only then invest in **1st place (outrunner / VGG-16 tile)** and **2nd place (Lopuhin / UNet + regressor)**. Every trial below must record **what we tried**, **Kaggle result**, and **why we kept, dropped, or deferred**.

**How to update:** After each lab run + Kaggle submit, add one row to §2 and one line to §3 if the decision changes. Copy `log/<run_name>.csv` epoch notes into §2 Notes if useful.

**Related:** `docs/COURSE_REQUIREMENTS.md` (deadlines, grading, mandatory report/slide items), `docs/REPORT_OUTLINE.md` (report skeleton), `AGENTS.md` (lab commands), `log/*.csv` (local metrics, gitignored).

**Course alignment:** Experiments here feed **Experimental Results** + **ablations** in the report; **Kaggle screenshot** and **contribution table** are separate checklist items in `COURSE_REQUIREMENTS.md`.

---

## 1. Strategy and ceiling criteria

| Phase | Goal | Stop / “ceiling” signal |
|-------|------|-------------------------|
| **Now** | Tile CNN: Inception (+ EffNet ablations), TTA, test scale, pup×1.2, h256 head, scale aug, strong blends | ≥2–3 submits with **&lt;0.2 RMSE** gain vs best, or all items in §5 backlog for current path done |
| **Next** | 1st: VGG-16 300×300 tile recipe (outrunner); 2nd: UNet density + secondary regressor (Lopuhin) | Only after §1 ceiling; separate experiment IDs `vgg_*`, `unet_*` |

**Why not 1st/2nd first (documented):**

- Repo built as `timm` tile regressor; fastest wins were Lopuhin **dot CSV** + Asanakoy **Inception/inference**, not full Keras VGG or UNet rewrites.
- HSV/dot extraction (outrunner-style) was unstable; we use Lopuhin `coords-threeplusone-v0.4.csv`.
- `gaussian_dots` (step toward density maps) **failed** on LB; full UNet deferred (`AGENTS.md`: out of scope until plateau).

---

## 2. Trial registry (Kaggle + local)

Fill **Private / Public** from `kaggle competitions submissions`. Local val = image-level RMSE from `validate.py` when noted.

| ID | Date | Backbone / config | Labels | Train | Infer | Post | Private | Public | Decision |
|----|------|-------------------|--------|-------|-------|------|---------|--------|----------|
| v2 | — | ResNet-50, area fraction | image area | 30 ep, bs128, t299 | full Test | — | 28.09 | 29.10 | Baseline; bad LB |
| v4 | — | v2 ckpt | area | (frozen) | Test×0.5, TTA, shifts=5 | pup×1.2 | 25.54 | 26.55 | **Keep:** test scale + pup cheap gain |
| v5 | — | ResNet-50 | balanced_dots | 20 ep | Test×0.5, TTA | pup×1.2 | 17.41 | 17.93 | **Keep:** dot supervision breakthrough |
| v6 | — | ResNet-50 | gaussian_dots | Phase 3 | Test×0.5, TTA | pup×1.2 | 27.40 | 28.40 | **Drop:** soft dots hurt |
| effnet_b3_v7 | — | EfficientNet-B3 | balanced_dots | 20 ep, bs128 | Test×0.5, TTA | pup×1.2 | 15.16 | 15.58 | **Keep:** strong blend candidate |
| **inception_v8** | — | Inception-ResNet v2 | balanced_dots | 20 ep, bs128 | Test×0.5, TTA | pup×1.2 | **14.58** | **14.47** | **Best production** |
| phase5 blend | — | v5 + v4 CSV | — | — | — | — | 17.61+ | — | **Drop:** weak ckpt hurts ensemble |
| inception_v8_test04 | (run) | v8 ckpt | — | — | Test×0.4 | pup×1.2 | _pending_ | _pending_ | Quick sweep, no retrain |
| inception_v9 | (run) | Inception + h256, scale 0.83–1.25 | balanced_dots | 35 ep | Test×0.5, TTA | pup×1.2 | _pending_ | _pending_ | Phase 7 Asanakoy medium scale |
| inception_v10 | (run) | Inception + h256, scale 0.66–1.5 | balanced_dots | 35 ep | Test×0.5, TTA | pup×1.2 | _pending_ | _pending_ | Phase 7 wide scale |
| blend v8+effnet | (run) | CSV ensemble | — | — | — | — | _pending_ | _pending_ | Only if both strong |

**Course target:** private RMSE **&lt; 13.05** (top-3 band). Historical Kaggle best ~**10.86**; golden ~15.88–10.86.

---

## 3. Decision log (why, not only what)

| # | Decision | Alternatives considered | Reason | Outcome |
|---|----------|----------------------|--------|---------|
| D1 | Tile regression vs whole-image resize | Classify whole image | Matches top Kaggle recipes; preserves resolution | Kept |
| D2 | Import Lopuhin dots vs `data.dots` HSV | Hand HSV, outrunner blob pipeline | HSV unstable; Lopuhin CSV gate &lt;1% vs train.csv | v5 −48% vs v2 |
| D3 | `balanced_dots` vs random tiles only | More background sampling | Empty tiles dominated; dot-centered crops | 17.41 |
| D4 | Skip full Lopuhin UNet (for now) | UNet + 2nd regressor | Large rewrite; dots alone gave most gain | Deferred → §5 |
| D5 | Skip outrunner VGG (for now) | VGG-16 300px Keras | Existing PyTorch/timm path; Inception beat ResNet after dots | Deferred → §5 |
| D6 | Test×0.5 + 5-shift TTA | Full-res test | Asanakoy / AGENTS; cheaper infer | v4, v8 gains |
| D7 | pup×1.2 post-process | None | outrunner tip; ~0.1–0.2 RMSE on v5 | Kept all submits |
| D8 | Inception over ResNet after dots | Stay on ResNet-50 | Same labels, −2.8 private RMSE | 14.58 |
| D9 | No ensemble with weak v4/v6 | Blend everything | Weak models pull RMSE up | Drop blends w/ weak |
| D10 | Phase 7: h256 + scale aug | Retrain VGG first | Fits `train.py`; Asanakoy missing pieces | Running |
| D11 | Document all trials in this file | Ad-hoc notes only | Repro + report + post-ceiling plan | This file |

_Add a row per major fork (e.g. new test scale, new blend weight, failed run)._

---

## 4. Per-run checklist (copy for new trials)

```text
ID:
Hypothesis:
Command / script:
Change vs previous best (one sentence):
Local val RMSE:
Kaggle private / public:
Keep | Drop | Iterate:
Notes for report:
```

---

## 5. Backlog after current pipeline plateaus

**Prepared (docs + scripts):** `docs/PHASE8_OUTRUNNER_VGG.md`, `docs/PHASE9_LOPUHIN_UNET.md`, `scripts/run_phase8_outrunner_vgg.sh`, `scripts/setup_lopuhin_vendor.sh`, `scripts/run_phases_8_and_9.sh`

**1st place — outrunner (VGG-16)** → Phase 8

- [ ] Run `bash scripts/run_phase8_outrunner_vgg.sh` → `outrunner_vgg300_pup120`
- [ ] `timm` `vgg16`, **300×300** tiles, `balanced_dots` (Lopuhin CSV)
- [ ] Optional: SGD / staged unfreeze (see PHASE8 doc); scale aug v8b
- [ ] pup×1.2 (via `finish_phase_run.sh`)

**2nd place — Lopuhin (UNet + regressor)** → Phase 9

- [ ] `bash scripts/setup_lopuhin_vendor.sh`
- [ ] UNet train + `make_submission.py` (vendor or modern port — see PHASE9 doc)
- [ ] Submit `lopuhin_unet_pup120`; blend with best tile CNN if strong

**Still on current path (before backlog)**

- [ ] Finish v9, v10, quick sweeps (test×0.4, blends)
- [ ] Strong-only ensembles (inception_v8 + effnet_b3_v7 ± v9/v10)
- [ ] Optional: more test scales (0.45, 0.55), shift count, epochs

---

## 6. References (leaderboard sources)

| Rank | Source | We use |
|------|--------|--------|
| 1st | [outrunner / yan-roo Keras VGG](https://github.com/yan-roo/Kaggle_Sea-Lions-Counting) | Idea + pup×1.2 only (so far) |
| 2nd | [lopuhin/kaggle-lions-2017](https://github.com/lopuhin/kaggle-lions-2017) | Dot CSV; UNet later |
| 4th | [asanakoy/kaggle_sea_lions_counting](https://github.com/asanakoy/kaggle_sea_lions_counting) | Inception, TTA, scale aug, test downscale |
