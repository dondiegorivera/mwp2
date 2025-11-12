# Market Prediction Workbench — Technical Overview

## Project Overview

- Purpose: End-to-end workbench for equity market forecasting using deep time-series models. It ingests raw OHLCV-like data, engineers targets and features, filters a tradable universe, trains a Temporal Fusion Transformer (TFT) with PyTorch Lightning, evaluates prediction quality (including calibrated uncertainty bands), and audits errors/outliers.
- Main functionality: Data preprocessing to Parquet, universe selection, label cleaning/winsorization, Hydra-configured training with rank-IC early stopping, checkpointing and W&B logging (optional), evaluation to predictions.csv plus metrics and figures, CV utilities, and auditing.
- Technology stack:
  - Languages: Python 3.10–3.13
  - ML/DL: PyTorch, PyTorch Lightning, PyTorch Forecasting (TFT)
  - Data: Polars, Pandas, PyArrow
  - Config: Hydra/OmegaConf
  - Viz/Tracking: Matplotlib, Weights & Biases (optional)
  - Build: Poetry
  - Environment: CUDA 12.1 wheels source configured (cu121)
- Architecture pattern: Monolithic, config-driven ML pipeline with CLI entry points (training, evaluation, CV, audits). No service/microservice components.

## System Architecture

- High-level diagram (conceptual):
  - Raw CSV → Data Pipeline (clean, map IDs, reindex, features, targets) → Processed Parquet
  - Processed Parquet → Universe Filter + Target Cleanup/ Winsorization → Model-ready Parquet
  - Model-ready Parquet → Training (Hydra-configured GlobalTFT) → Checkpoints/Logs
  - Checkpoints + Model-ready Parquet → Evaluation → predictions.csv, metrics, plots
  - predictions.csv + Processed Parquet → Audits/Outliers, Analysis CSVs

- Data flow and component interactions:
  - Data prep: `load_and_clean_data` → `create_mappings` → `reindex_and_fill_gaps` → `create_features_and_targets` (src/market_prediction_workbench/data.py)
  - Target cleanup and universe construction: scripts `clean_targets.py`, `filter_universe.py`, `winsorize_targets.py`
  - Training: `train.py` consumes processed Parquet via Pandas/Polars, constructs a PyTorch Forecasting `TimeSeriesDataSet`, builds a TFT wrapper (`GlobalTFT`), and fits with PL `Trainer`.
  - Evaluation: `evaluate.py` loads best checkpoint, reconstructs dataset parameters, runs inference and metrics, and writes artifacts.
  - CV and analysis: scripts `rolling_cv.py` (Hydra) and `tscv.py` (argparse) orchestrate multi-fold runs; `audit_outliers.py` and `analyze_preds.py` inspect predictions and residuals.

- External dependencies and integrations:
  - PyTorch ecosystem (TFT via PyTorch Forecasting), Hydra for config composition, optional Weights & Biases logger, filesystem-based artifacts (Parquet, CSV, PNG).

- Database schema overview:
  - No DB. Parquet files under `data/processed/` contain engineered columns: identifiers (ticker_id, sector_id), calendar and derived features, context features (market/sector/region returns and vols), engineered targets (`target_1d`, `target_5d`, `target_20d`), and `time_idx` per ticker.

## Directory Structure

- `src/market_prediction_workbench/` — core package
  - `data.py` — data pipeline and custom dataset
  - `model.py` — TFT wrapper and Lightning module
  - `train.py` — Hydra-configured training entry point
  - `evaluate.py` — checkpoint selection, inference, metrics, plots
  - `analyze_preds.py` — post-hoc analysis of predictions
- `scripts/` — CLI utilities (CV, winsorization, universe filtering, audits)
- `conf/` — Hydra configs (project config, model defaults, trainer defaults)
- `tests/` — unit/integration tests
- `data/` — raw and processed artifacts (not committed)
- `experiments/` — training/evaluation outputs
- `outputs/`, `lightning_logs/` — additional outputs/logs
- `pyproject.toml`, `poetry.lock` — build/deps
- `.pre-commit-config.yaml` — lint hooks

## Core Components Analysis

1) File path and name: `src/market_prediction_workbench/data.py:1`
- Primary responsibility: End-to-end data preparation for time-series forecasting, including cleaning, mapping IDs, reindexing to calendar with gap handling, feature engineering, and target construction. Provides a simple `MarketDataset` for sliding-window samples (used in tests) in addition to PF usage in training.
- Key functions/methods:
  - `load_and_clean_data(csv_path)` — read CSV via Polars, normalize schema, coerce types
  - `create_mappings(df, output_dir)` — produce `ticker_map.parquet` and `industry_map.parquet`; join IDs
  - `reindex_and_fill_gaps(df, max_ffill_days=5)` — per-ticker calendar upsample, mark `is_missing`, forward-fill limited windows
  - `create_features_and_targets(df)` — compute daily features (calendar encodings, RSI/MACD/vol, cross-sectional context metrics: market/sector/region) and trading-day targets at 1/5/20D; forward-fill non-label features; drop Null rows; assign `time_idx`
  - `MarketDataset` — Torch Dataset for fixed-length lookbacks with separation of known/unknown reals and static categoricals
- Interfaces exposed: Functional API above; dataset yields dict with tensors (`x_cat`, `x_known_reals`, `x_unknown_reals`, `y`, `groups`, `time_idx_window`).
- Dependencies: Polars, Torch, Dataclasses, Math
- Dependents: tests (`tests/test_data.py`, `tests/test_model.py`), pre-training pipeline, scripts referencing processed Parquet, training (indirectly via Parquet).

2) File path and name: `src/market_prediction_workbench/model.py:1`
- Primary responsibility: LightningModule `GlobalTFT` wrapping PyTorch Forecasting’s TFT, normalizing loss/quantile handling, device-safe mask building, and optimizer/scheduler wiring.
- Key functions/methods:
  - `TemporalFusionTransformerWithDevice.get_attention_mask(...)` — constructs decoder causal mask and encoder-zero mask on model’s device to avoid CPU/GPU mismatch.
  - `GlobalTFT.__init__(...)` — builds `TimeSeriesDataSet` from parameters or uses provided instance; configures loss and `output_size` from quantiles/targets; saves clean hparams; builds TFT via `from_dataset`.
  - `on_fit_start()` — attaches trainer/log proxies to inner model
  - `configure_optimizers()` — AdamW and schedulers: one-cycle or cosine-with-warmup; respects steps/epochs
- Interfaces exposed: `GlobalTFT` (LightningModule) instantiated by training/evaluation; forward delegated to underlying TFT.
- Dependencies: Torch, PL, PF (TFT internals), Numpy, Pandas, OmegaConf
- Dependents: `train.py`, `evaluate.py`, `scripts/rolling_cv.py`

3) File path and name: `src/market_prediction_workbench/train.py:1`
- Primary responsibility: Hydra entry point for training — loads processed Parquet, builds PF `TimeSeriesDataSet` with encoders/normalizers, constructs `GlobalTFT`, configures loaders (with optional `WeightedRandomSampler`), callbacks (EarlyStopping, LR monitor, val rank-IC), W&B logger (optional), and runs PL Trainer.
- Key functions/methods:
  - `get_embedding_sizes_for_tft(ds)` — compute categorical embeddings from PF encoders
  - dataset construction (group_ids, static/known/unknown reals, horizons, normalizers)
  - `RankICCallback` — computes daily Spearman rank-IC on validation predictions
  - Trainer/callbacks configuration and WandB setup
- Interfaces exposed: CLI via Hydra (`python src/market_prediction_workbench/train.py ...`)
- Dependencies: Hydra, PL, PF, Sklearn (optional scaler), Numpy/Pandas, Torch, OmegaConf
- Dependents: CLI users, CV scripts invoking training

4) File path and name: `src/market_prediction_workbench/evaluate.py:1`
- Primary responsibility: Select best checkpoint, reconstruct dataset from saved params, run batched inference, de-normalize per-group safely, compute metrics (MAE, RMSE, coverage, rank-IC), calibrate bands, and write artifacts.
- Key functions/methods:
  - `run_inference(model, loader, cfg, dataset)` — forward pass with robust de-normalization
  - `evaluate(preds, trues, names)` — MAE, RMSE, 90% interval coverage; quantile index resolution from model loss
  - Calibration helpers: alpha estimation from residuals and re-computation of calibrated intervals
- Interfaces exposed: Hydra CLI entry point; writes `experiments/evaluation/<run>/predictions.csv`, `metrics.json`, figures
- Dependencies: Hydra, Torch, PF, Numpy, Pandas, Polars, Matplotlib, SciPy (spearmanr)
- Dependents: `scripts/audit_outliers.py`, `analyze_preds.py`

5) File path and name: `scripts/rolling_cv.py:555`
- Primary responsibility: Hydra-driven rolling CV over contiguous validation windows with embargo; per-fold train/eval artifacts; fold aggregation with rank-IC and costed Sharpe.
- Interfaces: CLI `python scripts/rolling_cv.py` with Hydra overrides
- Dependencies: Hydra, PL/PF, Numpy/Pandas, Torch, SciPy
- Dependents: Experiment management

6) File path and name: `scripts/tscv.py:1`
- Primary responsibility: Simple time-series CV runner using argparse; orchestrates multiple train runs with date-based splits.
- Interfaces: CLI with `--processed`, `--folds`, `--embargo`, `--epochs`
- Dependencies: Pandas/Numpy, subprocess

7) File path and name: `scripts/winsorize_targets.py:1`, `scripts/clean_targets.py:1`, `scripts/filter_universe.py:1`
- Primary responsibility: Label shaping and universe selection utilities
  - winsorize (per-ticker caps inferred on train slice), drop/clip extreme labels, filter to equities and liquidity thresholds; optional event-window removal from raw price moves
- Interfaces: CLI via argparse
- Dependencies: Pandas/Numpy/Polars
- Dependents: Pre-training data pipeline, `start.md` workflow

8) File path and name: `scripts/audit_outliers.py:1`, `src/market_prediction_workbench/analyze_preds.py:1`
- Primary responsibility: Post-eval audits — residual histograms, outlier tables, per-ticker/per-date breakdowns, coverage checks with preference for CSV-provided ground truth columns.
- Interfaces: CLI
- Dependencies: Pandas/Numpy/Matplotlib, Polars (optional), JSON
- Dependents: Offline analysis of latest evaluation run

9) File path and name: `conf/config.yaml:1`, `conf/model/tft_default.yaml:1`, `conf/trainer/default_trainer.yaml:1`
- Primary responsibility: Central configuration for data, model, and trainer with sensible defaults, including cosine warmup LR schedule, precision, accumulate grad, early stopping by rank-IC, batch sizes, and optional W&B.
- Interfaces: Composed by Hydra in train/eval/CV
- Dependencies: Hydra runtime
- Dependents: All Hydra entry points

## Entry Points

- Main application entry points:
  - Training: `src/market_prediction_workbench/train.py:295`
  - Evaluation: `src/market_prediction_workbench/evaluate.py:570`
  - Rolling CV: `scripts/rolling_cv.py:555`
  - TSCV: `scripts/tscv.py:1`
  - Audits/Analysis: `scripts/audit_outliers.py:1`, `src/market_prediction_workbench/analyze_preds.py:1`

- Build/deployment configurations:
  - Poetry-managed at project root (`pyproject.toml`). CUDA 12.1 wheel index configured.
  - No Docker/CI committed; local or scripted orchestration.

- Environment requirements:
  - Python >=3.10,<3.14, PyTorch + CUDA 12.1 compatible GPU (optional but recommended)
  - `poetry install` to resolve dependencies. Optional W&B credentials if enabled.

## API/Interface Documentation

- REST/GraphQL: None
- Internal module interfaces:
  - Data pipeline functions and `MarketDataset` outputs (see data component)
  - `GlobalTFT` LightningModule with standard PL interface for Trainer
  - PF `TimeSeriesDataSet` parameters flow through dataset construction and model `from_dataset`
- Event handlers/listeners:
  - PL callbacks: EarlyStopping on `val_rank_ic`, LearningRateMonitor; custom `RankICCallback` computes daily rank-IC during validation.

## Data Models

- DataConfig dataclass: `src/market_prediction_workbench/data.py:11`
  - `static_reals`, `static_categoricals`, `time_varying_known_reals`, `time_varying_unknown_reals`, `target_columns`, `lookback_days`, `prediction_horizon`

- DataFrames and schema:
  - Identifiers: `ticker`, `ticker_id`, `industry`, `sector_id`, optional `region`
  - Calendar: `date`, derived `day_of_week`, `day_of_month`, `month`, `is_quarter_end`, `time_idx`
  - Observed features: `log_return_1d/5d/20d`, `volume_zscore_20d`, `volatility_20d`, `rsi_14d`, `macd`, etc.
  - Context features: `mkt_ret_1d`, `mkt_vol_20d`, `sector_ret_1d`, `sector_vol_20d`, `region_ret_1d`, `region_vol_20d`
  - Targets: `target_1d`, `target_5d`, `target_20d` on trading-day sequences

- Validation rules:
  - Null handling: forward-fill non-label features per-ticker; final `.drop_nulls()` to ensure all required fields present
  - Guardrails for NaN/Inf → Null conversion for float columns
  - Lookback window consistency checks in `MarketDataset`

## State Management

- No client-side state libraries. Runtime state handled by:
  - Hydra configs (composition and overrides)
  - PL Trainer/Callback state and checkpointing
  - Filesystem artifacts for datasets, models, and evaluation outputs
  - Optional W&B run state

## Current Technical Debt & Issues

- Hydra defaults reference a missing group: `conf/config.yaml:1` has `- data: default_data_config` but `conf/data/default_data_config.yaml` is absent. Either add the file or remove the default and keep `data:` inline.
- Tests: `tests/test_model.py` uses `pl` without importing Polars; this will raise `NameError` if executed in isolation. Add `import polars as pl`.
- Dual Lightning packages: both `pytorch-lightning` and `lightning` are declared; ensure intentional and compatible.
- Private import from PF: `pytorch_forecasting.models.temporal_fusion_transformer._tft` (leading underscore) can change across PF releases; consider stable API imports.
- Mask/device logic: custom attention mask in `TemporalFusionTransformerWithDevice` is critical; keep aligned with PF upstream if upgrading.
- Pre-commit: Black and Ruff-format are disabled; code style may drift.
- Artifacts layout: multiple experiment output dirs (`experiments/`, `lightning_logs/`, W&B) — consider consolidation and cleanup policy.
- Legacy or confusing dirs: `market-prediction-workbench/` directory exists but is not the importable package (actual code is under `src/market_prediction_workbench/`). Clarify or remove if unused.

## Testing Infrastructure

- Frameworks: PyTest
- Tests present:
  - `tests/test_data.py` — integration of data pipeline and `MarketDataset`
  - `tests/test_model.py` — builds a tiny PF dataset and exercises forward pass + a sampler test
  - `tests/test_audit_coverage.py` — verifies coverage metric respects explicit truth columns
- Coverage summary: Minimal to moderate; many paths (Hydra config matrix, evaluation/calibration branches, CV) not covered.
- Gaps: No tests for training loop hooks, scheduler wiring, W&B logger flow, evaluate best-ckpt selection, or CV pipelines end-to-end.

## Build & Deployment

- Build process:
  - `poetry install` (Python >=3.10,<3.14). CUDA 12.1 wheel index configured for Torch.
  - Optional: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for stability.

- CI/CD: Not present. Recommend adding GitHub Actions or similar for lint/test.

- Deployment targets: Local training/evaluation; no serving component.

- Configuration management:
  - Hydra configs under `conf/` with overrides via CLI (e.g., `trainer.epochs=15`). `start.md` documents typical pipelines and overrides.

## Gaps & Improvement Opportunities

- Refactoring
  - Extract stable import paths for TFT (avoid private `_tft`) and encapsulate mask logic behind feature flags to ease upgrades.
  - Modularize sampler creation logic in `train.py` for testability and reuse (currently embedded with several fallbacks).
  - Consolidate calibration utilities (duplicated patterns across evaluate/analyze/audit).

- Documentation
  - Add `conf/data/default_data_config.yaml` or remove the default include; document all config keys in README.
  - Expand README with end-to-end quickstart (data prerequisites, commands, expected outputs) mirroring `start.md`.

- Additional features
  - Add `rolling_cv.py` result aggregation visualizations and convenience scripts for comparing runs.
  - Add optional macro ingestion recipe and schema in `data/external/macro.parquet` (currently best-effort join).

- Performance optimization
  - Profile dataloaders with and without `WeightedRandomSampler` and re-balance strategies by group/time to minimize variance and maximize throughput.
  - Consider Polars → PyTorch zero-copy or memory-mapped pipelines where possible.
  - Ensure mixed-precision (`bf16-mixed`/`16-mixed`) choices align with GPU/driver for stability.

- Security hardening
  - Sanitize path handling and user-provided overrides in CLI tools; validate numeric ranges for CV/window parameters.

- Modernization
  - Adopt a single Lightning package (prefer the maintained one) and align versions with PF/Torch matrix.
  - Enable formatting in pre-commit (Black or Ruff-format) and add Ruff rule set tuned for this project.
  - Add basic CI (lint + unit tests + smoke training on tiny data).

## Actionable Next-Sprint Items

1) Fix configuration and test issues
- Add `conf/data/default_data_config.yaml` or remove default; update README accordingly.
- Add `import polars as pl` to `tests/test_model.py` and verify tests run with a small synthetic dataset.

2) Stabilize dependencies
- Decide on Lightning package (`lightning` vs `pytorch-lightning`) and align versions with PF/Torch; replace private TFT import with public API if available.

3) Improve evaluation/calibration cohesion
- Centralize calibration logic and coverage metrics in a shared module; ensure evaluate/analyze/audit consume the same helpers.

4) Add CI and formatting
- Introduce GitHub Actions to run Ruff (lint), Black (format), and PyTest (data-free units + tiny synthetic fixtures). Re-enable formatter hooks.

5) Sampler and metrics robustness
- Unit-test the sampler weighting and fallback paths; add tests for rank-IC callback and LR schedule configuration.

6) Documentation & examples
- Extend `README.md` with: data schema, feature list, config reference, and a full reproducible training/eval flow using a tiny demo dataset.
