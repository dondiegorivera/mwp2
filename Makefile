# Basic workflow targets for the Market Prediction Workbench.

POETRY ?= poetry
PYTHON ?= $(POETRY) run python

RAW_PARQUET ?= data/processed/processed_data.parquet
CLEAN_PARQUET ?= data/processed/processed_data.cleaned.parquet
EQONLY_PARQUET ?= data/processed/processed_data.eqonly.parquet
WIN_PARQUET ?= data/processed/processed_data.eqonly.win.parquet
TICKER_MAP ?= data/processed/ticker_map.parquet
RUN_DIR ?= experiments/evaluation/latest

.PHONY: help install data-pipeline clean-targets filter-universe winsorize \
	start-train start-eval start-cv start-audit start-analyze \
	lint format format-check pre-commit test test-fast test-sampling

help:
	@echo "Common targets:"
	@echo "  make install          # Install dependencies via Poetry"
	@echo "  make data-pipeline    # Run the end-to-end data pipeline"
	@echo "  make start-train      # Launch training (override TRAIN_ARGS=...)"
	@echo "  make start-eval       # Evaluate the latest checkpoint"
	@echo "  make start-cv         # Run rolling cross-validation (override CV_ARGS=...)"
	@echo "  make start-audit      # Audit predictions (set RUN_DIR=...)"
	@echo "  make start-analyze    # Analyze predictions.csv"
	@echo "  make test             # Run full pytest suite"
	@echo "  make test-fast        # Run lightweight, data-free tests"

install:
	$(POETRY) install

data-pipeline: data ingest clean-targets filter-universe winsorize

data:
	$(PYTHON) src/market_prediction_workbench/data.py

clean-targets:
	$(PYTHON) scripts/clean_targets.py --in $(RAW_PARQUET) --out $(CLEAN_PARQUET) --train-cutoff 0.8 --mode drop

filter-universe:
	$(PYTHON) scripts/filter_universe.py --in $(CLEAN_PARQUET) --ticker-map $(TICKER_MAP) --out $(EQONLY_PARQUET) --auto-target-tickers 800

winsorize:
	$(PYTHON) scripts/winsorize_targets.py --in $(EQONLY_PARQUET) --out $(WIN_PARQUET) --train-cutoff 0.8

start-train:
	$(PYTHON) src/market_prediction_workbench/train.py $(TRAIN_ARGS)

start-eval:
	$(PYTHON) src/market_prediction_workbench/evaluate.py $(EVAL_ARGS)

start-cv:
	$(PYTHON) scripts/rolling_cv.py $(CV_ARGS)

start-audit:
	$(PYTHON) scripts/audit_outliers.py --run-dir $(RUN_DIR) --data-parquet $(RAW_PARQUET) --ticker-map $(TICKER_MAP) $(AUDIT_ARGS)

start-analyze:
	$(PYTHON) src/market_prediction_workbench/analyze_preds.py $(ANALYZE_ARGS)

lint:
	$(POETRY) run ruff check .

format:
	$(POETRY) run black .

format-check:
	$(POETRY) run black --check .

pre-commit:
	$(POETRY) run pre-commit run --all-files

test:
	$(POETRY) run pytest -q

test-fast:
	$(POETRY) run pytest tests/test_sampling_and_training.py -q

test-sampling:
	$(POETRY) run pytest tests/test_sampling_and_training.py -q
