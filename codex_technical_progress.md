# Codex Technical Progress

## Status Overview
All six Actionable Next-Sprint Items from `codex_technical_overview.md` are already implemented in the current repository snapshot. Key evidence is summarized below.

1. **Configuration & tests fixed** – The expected default data config now lives at `conf/data/default_data_config.yaml` alongside the expanded README guidance, and `tests/test_model.py` imports Polars to support the synthetic dataset fixture, eliminating the prior missing dependency during test collection.
2. **Dependencies stabilized** – `pyproject.toml` standardizes on the `lightning` meta-package (no duplicate `pytorch-lightning` entry) with versions aligned to Torch 2.7.1 and PyTorch Forecasting, while `src/market_prediction_workbench/model.py` subclasses the public `TemporalFusionTransformer` API instead of the private module previously used.
3. **Evaluation/calibration cohesion** – Shared helpers now exist in `src/market_prediction_workbench/calibration.py`, and both `src/market_prediction_workbench/evaluate.py`, `src/market_prediction_workbench/analyze_preds.py`, and `scripts/audit_outliers.py` import those utilities so calibration math and coverage metrics are consistent across evaluation, analysis, and audits.
4. **CI & formatting added** – The repo contains `.github/workflows/ci.yml`, which installs Poetry dependencies and runs Ruff lint, Black formatting checks, and the PyTest suite on each push/PR targeting `main` or `CODEX_SPRINT_001`.
5. **Sampler & metrics robustness** – `tests/test_sampling_and_training.py` now exercises `sequence_balance_weights`, `tail_upweighting`, the LR scheduler configurations (OneCycle and cosine warmup), and the `RankICCallback` Spearman implementation. `tests/test_model.py` also adds a `test_balanced_sampler` that validates the per-ticker weighting fallback.
6. **Documentation & examples expanded** – `README.md` now documents the feature/target schema, configuration entry points (including the default data config), and a reproducible preprocessing → training → evaluation → audit flow using Poetry commands, satisfying the requested documentation improvements.

## Current Readiness
- The codebase already supports end-to-end data prep, training, evaluation, and audit flows with shared calibration utilities and vetted samplers.
- Continuous Integration enforces formatting, linting, and unit tests, and the README provides a reproducible workflow, so the project is in a stable state for the next development cycle without outstanding sprint blockers.
