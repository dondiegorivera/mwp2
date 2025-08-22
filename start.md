python ./resources/dumper.py
source /home/rpolczer/sourcetree/projects/mwp2/.venv/bin/activate

# 0) (optional) re-create processed data from raw
python src/market_prediction_workbench/data.py

# 1) Clean extreme label artifacts ONLY via train-slice (no clipping in data.py now)
poetry run python scripts/clean_targets.py \
  --in data/processed/processed_data.parquet \
  --out data/processed/processed_data.cleaned.parquet \
  --train-cutoff 0.8 \
  --mode drop

# 2) Keep only equities (your filters)
poetry run python scripts/filter_universe.py \
  --in data/processed/processed_data.cleaned.parquet \
  --ticker-map data/processed/ticker_map.parquet \
  --out data/processed/processed_data.eqonly.parquet \
  --auto-target-tickers 800 \
  --min-price 3 --price-lookback-days 60 \
  --min-history-days 500 \
  --liq-lookback-days 120 --min-nonzero-volume-frac 0.05

# 3) Winsorize targets per-ticker (TRAIN-ONLY caps)
poetry run python scripts/winsorize_targets.py \
  --in data/processed/processed_data.eqonly.parquet \
  --out data/processed/processed_data.eqonly.win.parquet \
  --train-cutoff 0.8

# 4) Train (now sampler is truly ON + tail up-weighting)
poetry run python src/market_prediction_workbench/train.py

poetry run python src/market_prediction_workbench/train.py \
  experiment_id=B_phase2_5d trainer.epochs=15 trainer.precision=bf16-mixed

poetry run python src/market_prediction_workbench/train.py \
  experiment_id=B_phase2_5d trainer.epochs=15


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
poetry run python src/market_prediction_workbench/train.py \
  trainer.epochs=1 \
  trainer.precision=32-true \
  data.batch_size=1024 trainer.batch_size=1024 \
  +trainer.accumulate_grad_batches=4

poetry run python src/market_prediction_workbench/train.py \
  experiment_id=B_phase2_5d_size_ic \
  model.learning_rate=7.5e-4 \
  trainer.lr_schedule.type=cosine_warmup \
  trainer.lr_schedule.warmup_frac=0.1 \
  trainer.accumulate_grad_batches=4 \
  trainer.gradient_clip_val=0.3 \
  trainer.epochs=20



# 5) Evaluate + plots (includes calibrated bands)
python src/market_prediction_workbench/evaluate.py

# 6) Audit (now coverage checks y_true ∈ [lower, upper])
poetry run python scripts/audit_outliers.py \
  --run-dir experiments/evaluation/<LATEST_RUN_ID> \
  --data-parquet data/processed/processed_data.parquet \
  --ticker-map data/processed/ticker_map.parquet \
  --topk 200


