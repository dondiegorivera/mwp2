python ./resources/dumper.py
source /home/rpolczer/sourcetree/projects/mwp2/.venv/bin/activate


python src/market_prediction_workbench/data.py

poetry run python scripts/clean_targets.py   --in data/processed/processed_data.eqonly.parquet   --out data/processed/processed_data.eqonly.cleaned.parquet   --train-cutoff 0.8   --mode drop

poetry run python scripts/filter_universe.py \
  --in data/processed/processed_data.cleaned.parquet \
  --ticker-map data/processed/ticker_map.parquet \
  --out data/processed/processed_data.eqonly.parquet

poetry run python scripts/winsorize_targets.py \
  --in data/processed/processed_data.eqonly.parquet \
  --out data/processed/processed_data.eqonly.win.parquet \
  --train-cutoff 0.8


poetry run python src/market_prediction_workbench/train.py
python src/market_prediction_workbench/evaluate.py

poetry run python scripts/audit_outliers.py   --run-dir experiments/evaluation/jeus0uzq   --data-parquet data/processed/processed_data.parquet   --ticker-map data/processed/ticker_map.parquet   --topk 200