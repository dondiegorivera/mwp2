# market-prediction-workbench

Market Prediction Workbench for stock forecasting.

## Setup

1.  Clone the repository.
2.  Ensure you have Poetry installed.
3.  Install dependencies:
    ```bash
    poetry install
    ```
4.  Activate the virtual environment:
    ```bash
    poetry shell
    ```
5.  Set up pre-commit hooks (if not already done by someone else who committed `.git/hooks`):
    ```bash
    poetry run pre-commit install
    ```

## Project Structure

-   `data/`: Raw and processed data.
-   `notebooks/`: EDA and scratch work.
-   `src/market_prediction_workbench/`: Core Python package.
-   `conf/`: Configuration files.
-   `experiments/`: Experiment logs and outputs.
-   `tests/`: Unit tests.

## Data Schema & Features

- **Identifiers**: `ticker`, `ticker_id`, `sector_id`, `date`, `time_idx`.
- **Targets**: `target_1d`, `target_5d`, `target_20d` (trading-day log returns).
- **Calendar features**: `day_of_week`, `day_of_month`, `month`, `is_quarter_end`.
- **Technical signals**: `log_return_*`, `volume_zscore_20d`, `volatility_20d`, `rsi_14d`, `macd`, `market_cap_static_norm`.
- **Context features**: `mkt_ret_1d`, `sector_ret_1d`, `region_ret_1d` with 20d vol counterparts.
- The full list lives in `src/market_prediction_workbench/data.py`. Run the script to regenerate processed parquet from raw CSVs.

## Configuration

- Hydra drives configuration. Base defaults live under `conf/`.
- `conf/data/default_data_config.yaml` defines data-specific knobs (group IDs, target columns, horizons, feature lists).
- Override values on the CLI, e.g.:
  ```bash
  poetry run python src/market_prediction_workbench/train.py data.target='["target_1d","target_5d"]'
  ```

## Training & Evaluation Quickstart

1. Preprocess data (see `start.md` for the full pipeline). Minimal example:
   ```bash
   poetry run python src/market_prediction_workbench/data.py
   poetry run python scripts/clean_targets.py --in data/processed/processed_data.parquet --out data/processed/processed_data.cleaned.parquet
   poetry run python scripts/filter_universe.py --in data/processed/processed_data.cleaned.parquet --ticker-map data/processed/ticker_map.parquet --out data/processed/processed_data.eqonly.parquet
   poetry run python scripts/winsorize_targets.py --in data/processed/processed_data.eqonly.parquet --out data/processed/processed_data.eqonly.win.parquet
   ```
2. Train:
   ```bash
   poetry run python src/market_prediction_workbench/train.py trainer.epochs=5
   ```
3. Evaluate the latest checkpoint (writes predictions/metrics under `experiments/evaluation/<run>`):
   ```bash
   poetry run python src/market_prediction_workbench/evaluate.py
   ```
4. Audit and analyze predictions:
   ```bash
   poetry run python scripts/audit_outliers.py --run-dir experiments/evaluation/<RUN> --data-parquet data/processed/processed_data.parquet --ticker-map data/processed/ticker_map.parquet
   poetry run python src/market_prediction_workbench/analyze_preds.py
   ```

All commands assume `poetry install` has been run and the processed parquet files exist at the configured paths.
