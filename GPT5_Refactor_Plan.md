1. **Overview (5–10 lines max)**

We’ll harden data hygiene (date-based split + embargo, no target dropping, safer calendar features), fix normalization pitfalls (don’t scale booleans; add cyclical encodings), restore balanced sampling, and make evaluation horizon-correct with robust inverse transforms. Changes stay inside existing files: `data.py`, `train.py`, `evaluate.py`, and `conf/config.yaml`; tests remain compatible.
**Parallel:** (a) config tweaks in `conf/config.yaml`; (b) evaluation metrics expansion in `evaluate.py`. Everything else runs **sequentially**.

---

2. **Steps (numbered, strictly ordered)**

```
Step 1: Add split & sampler knobs to config
Why:
- Centralize split gap/percent + enable/disable balanced sampling without code edits.

Targets:
- conf/config.yaml

Instructions:
- Under root, add:
  data:
    split:
      use_date_split: true
      train_pct: 0.8
      embargo_days: 30         # exclusion gap between train/val
  trainer:
    use_weighted_sampler: true
- Keep existing keys intact.

Tests:
- None (config-only).

Verify:
- hydra prints show the new keys in the config dump at train start.

Dependencies:
- None
```

```
Step 2: Switch to date-based split with embargo gap
Why:
- Prevent leakage from mixed per-ticker time_idx and overlapping windows.

Targets:
- src/market_prediction_workbench/train.py (main)

Instructions:
- After `data_pd` load/casts, compute:
  min_date = data_pd['date'].min(); max_date = data_pd['date'].max()
  cutoff_date = min_date + (max_date - min_date) * cfg.data.split.train_pct
  embargo = pd.Timedelta(days=cfg.data.split.embargo_days)
- Replace current `time_idx` split with:
  train_df = data_pd[data_pd['date'] <= (cutoff_date - embargo)]
  val_df   = data_pd[data_pd['date'] >= (cutoff_date + embargo)]
- Keep the rest (dataset_params, normalizers) unchanged.

Tests:
- Run existing tests (no changes); training still forms non-empty datasets.

Verify:
- Printed shapes show both splits >0; dates don’t overlap (check min/max of each).

Dependencies:
- Step 1
```

```
Step 3: Stop scaling booleans; add cyclical calendar features (keep old cols for compat)
Why:
- Avoid distorting binary flags; make time features rotationally consistent.

Targets:
- src/market_prediction_workbench/data.py (create_features_and_targets)
- src/market_prediction_workbench/train.py (reals scaler selection)

Instructions:
- In data.py, keep existing: day_of_week, day_of_month, month, is_quarter_end.
- Add:
  dow_sin = sin(2π * day_of_week/7); dow_cos = cos(2π * day_of_week/7)
  mon_sin = sin(2π * month/12);      mon_cos = cos(2π * month/12)
  (cast to Float32)
- Ensure is_quarter_end and is_missing remain 0/1 floats.
- In train.py, when building `reals_to_scale`, exclude:
  ['is_quarter_end','is_missing','day_of_week','day_of_month','month']
  (keep their raw values) — cyclical pairs (dow_*, mon_*) may be left unscaled or globally via EncoderNormalizer; do **not** GroupNormalize them.

Tests:
- Existing tests still pass (they reference old columns); new cyclical cols are additive.

Verify:
- Inspect a sample: booleans in {0.0,1.0}; cyclical values in [-1,1].

Dependencies:
- None
```

```
Step 4: Replace target dropping with winsorization (clip, don’t delete)
Why:
- Preserve tails for P&L; avoid sequence breaks and bias from row drops.

Targets:
- src/market_prediction_workbench/data.py (create_features_and_targets)

Instructions:
- Replace the block that sets `None` on targets + later `drop_nulls(subset=target_cols)` with:
  MAX_ABS_DAILY_RETURN = 0.25
  target_1d  = pl.col("target_1d").clip(-MAX_ABS_DAILY_RETURN, MAX_ABS_DAILY_RETURN)
  target_5d  = pl.col("target_5d").clip(-MAX_ABS_DAILY_RETURN*5, MAX_ABS_DAILY_RETURN*5)
  target_20d = pl.col("target_20d").clip(-MAX_ABS_DAILY_RETURN*20, MAX_ABS_DAILY_RETURN*20)
- Remove the subsequent `drop_nulls(subset=target_cols)` that depended on Nones.

Tests:
- `tests/test_data.py::test_dataset_item_shape_and_no_nans` should still see no NaNs.

Verify:
- No reduction in row count from target filtering; tail values are clipped not removed.

Dependencies:
- None
```

```
Step 5: Reinstate balanced sampling for tickers
Why:
- Prevent mega-cap histories from dominating batches.

Targets:
- src/market_prediction_workbench/train.py (train_loader creation)

Instructions:
- If `cfg.trainer.use_weighted_sampler`:
  • Compute per-row weights in `train_df`: w = 1 / count(ticker_id), normalize optional.
  • Create `torch.utils.data.WeightedRandomSampler(weights=w, num_samples=len(training_dataset), replacement=True)`.
  • Pass `sampler=` to `training_dataset.to_dataloader(train=True, ...)` and set `shuffle=False`.
- Keep val_loader as is.

Tests:
- `tests/test_model.py::test_balanced_sampler` conceptually aligns; no change needed.

Verify:
- Print a small histogram of sampled group ids for sanity (optional log).

Dependencies:
- Step 2 (train_df exists)
```

```
Step 6: Harden feature rolling against synthetic bars
Why:
- Ensure rolling stats don’t bleed across prolonged missing sequences.

Targets:
- src/market_prediction_workbench/data.py (create_features_and_targets)

Instructions:
- Reuse existing `is_missing` mask (already present).
- Ensure every rolling/ewm expression is wrapped with `pl.when(~is_missing_mask)` *inside* the `.over("ticker_id")` (this is already done for key features; extend to any remaining rolling that isn’t).
- After forward-filling features later in the function, keep `is_missing` unchanged (don’t fill it).

Tests:
- Existing tests (NaN checks) remain valid.

Verify:
- Spot-check a long holiday span: rolling columns are null over missing, then ffilled per current step (bounded by earlier ffill limits).

Dependencies:
- None
```

```
Step 7: Use robust inverse-transform & evaluate per horizon
Why:
- Fix PF API/version quirks; report correct metrics for 1d/5d/20d.

Targets:
- src/market_prediction_workbench/evaluate.py (run_inference, evaluate)

Instructions:
- Keep `_inverse_with_groups` (already present).
- In `run_inference`, stop collapsing to horizon 0 only:
  • Remove `preds_h1 = preds_dec[:, 0]` and `trues_h1 = trues_dec[:, 0]`.
  • For each horizon h in [0..H-1], collect preds/trues into keys like:
    f"{name}@h{h+1}", f"{name}_lower@h{h+1}", f"{name}_upper@h{h+1}".
- In `evaluate(...)`, iterate names × horizons, computing MAE/RMSE/Coverage for each.
- Write horizon-tagged metrics to `metrics.json`; keep plotting the first horizon as before to avoid UI bloat.

Tests:
- None (eval script); quick manual inspection of saved `metrics.json`.

Verify:
- Printed metrics include `*_@h1`, `*_@h5`, `*_@h20` (depending on H).

Dependencies:
- None
```

```
Step 8: Guard categorical encoders & embeddings
Why:
- Avoid silent mis-embeddings when PF doesn’t create encoders for every categorical.

Targets:
- src/market_prediction_workbench/train.py (get_embedding_sizes_for_tft)

Instructions:
- Keep the current robust function.
- Before assigning `model_specific_params_from_cfg["embedding_sizes"]`, ensure it’s a dict of `{cat_name: (cardinality, dim)}`; if empty, log a warning and proceed (TFT will default).

Tests:
- Covered by `tests/test_model.py::test_tft_forward`.

Verify:
- Startup logs show calculated sizes per categorical.

Dependencies:
- None
```

```
Step 9: Preserve 0/1 flags through scaling pipeline
Why:
- Double-check booleans aren’t altered by global fills.

Targets:
- src/market_prediction_workbench/data.py (create_features_and_targets, final fill)
- src/market_prediction_workbench/train.py (scalers)

Instructions:
- In data.py’s “forward-fill features” block, exclude ['is_quarter_end','is_missing'] from ffill/fill_null lists so they remain exact 0/1 (leave them as originally computed).
- In train.py, ensure these columns are not added to `scalers`.

Tests:
- Add assertion inside `test_dataset_item_shape_and_no_nans`: (optional if you want) check unique of is_quarter_end ∈ {0.0,1.0}. If not modifying tests, manually verify.

Verify:
- Quick describe() shows min=0, max=1 for both flags.

Dependencies:
- Step 3
```

```
Step 10: Make W&B run reproducible & captured configs
Why:
- Ensure exact config/dataset parameters are stored with runs.

Targets:
- src/market_prediction_workbench/train.py (W&B block)

Instructions:
- Keep the existing Hydra `.hydra` copy code.
- Also log: number of train/val samples, min/max date of each split, and the final feature lists (categoricals, known/unknown reals) as W&B artifacts/summary.

Tests:
- None.

Verify:
- W&B run shows these fields in config or summary.

Dependencies:
- Steps 2–5
```

---

3. **Final checks (bullet list, ≤8 bullets)**

* Splits are by **date** with **embargo**, and train/val date ranges don’t overlap.
* Targets are **winsorized** (no row drops), sequences remain contiguous.
* **Booleans** (`is_quarter_end`, `is_missing`) remain 0/1 and are **not scaled**; cyclical features added.
* All rolling/ewm stats are masked with `~is_missing` (no bleed across synthetic bars).
* **Weighted sampler** active (if enabled) and train loader uses `sampler` with `shuffle=False`.
* Evaluation reports **per-horizon** metrics; inverse transform works across PF versions.
* W\&B captures config + split stats; checkpoints already store dataset params (unchanged).
* **Rollback:** revert Step 2 split block to prior `time_idx` split, disable sampler in config, and undo Step 4 (winsorization) by restoring prior target handling.
