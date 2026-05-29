# Pipeline audit (training / inference / submission)

Deep review date: 2026-05-29. Fixes applied in the same commit batch.

## P0 fixes applied

| Issue | Fix |
|-------|-----|
| **CRLF in `scripts/*.sh`** broke Linux (`set -euo pipefail`) | `sed` normalize + `.gitattributes` `*.sh text eol=lf` |
| **`run_infer_v5.sh` hardcoded `--stride 299`** | Removed; `inference.py` uses checkpoint `tile_size` |
| **Duplicate Kaggle submit** in quick sweeps | Removed extra `submit.sh` after `finish_phase_run.sh` |
| **`tee log/...` without `mkdir log`** | `mkdir -p log submission` in `run_infer_v5.sh` |
| **Missing checkpoint / test dir** | `fp_paths.sh` helpers + checks in infer/finish/phase1 |
| **Missing test images silently zero-filled** | `inference.py` now fails unless `--allow_missing_test_images` is explicit |
| **Checkpoint optimizer state loaded onto GPU for infer/validate** | Load checkpoints on CPU, then move model only |

## P1 fixes applied

| Issue | Fix |
|-------|-----|
| **Hard-coded v9/v10 ckpt paths** | `fp_checkpoint_path()` in `run_top3_train_queue.sh` |
| **`run_phase1` infer used `Test/` not scaled** | Default `TEST_SUBDIR=Test_scaled_0.5` + `--test_subdir` |
| **Double `torch.load` in inference** | Single load + `load_state_dict` |
| **Invalid `scale_min == scale_max`** | `train.py` raises clear error |
| **Phase 9-only run silent success** | `PHASE=9` exits 1 if no Lopuhin CSV |
| **Scale augmentation never zoomed in** | Source crop can now be smaller than `tile_size` when sampled scale > 1 |
| **Inference profiling double-counted tiles** | `InferenceTimings.add_image()` is now the only tile counter update |
| **Targeted `run_tests.sh` still collected all tests** | Script now respects explicit pytest arguments |
| **Env-overridden BS/RUN_NAME could desync checkpoint paths** | Phase scripts now pass exact backbone/epoch/batch/tile/suffix and clear inherited `RUN_NAME` |

## Guard script (run on lab after pull)

```bash
bash scripts/verify_shell_scripts.sh
bash scripts/run_tests.sh -q
```

## Known risks (document, not bugs)

- **Train val uses `val_shifts=1`; infer uses 5** — val RMSE ≠ deploy metric.
- **`run_phase1` default submit is raw CSV** (no pup×1.2); top-3 flows use `finish_phase_run.sh`.
- **Phase 9 Lopuhin** needs PyTorch 0.1 vendor env or a modern port (`docs/PHASE9_LOPUHIN_UNET.md`).
- **`--max_images` on inference** fills rest of submission with zeros; use only for profiling/debug.
- **Old checkpoints without `count_columns`** use legacy column order (`data/targets.py`).

## Pipeline contract (must hold)

1. **Infer stride defaults to checkpoint `tile_size`**; set `STRIDE` only for an intentional override.
2. **Same test subdir** as used in production (`Test_scaled_0.5` typical).
3. **`head_hidden` / `backbone` in checkpoint** → `build_counter_from_checkpoint_args`.
4. **Dot gate** before `balanced_dots` training (unless `--skip_dot_cache_gate`).
