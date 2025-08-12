python ./resources/dumper.py
source /home/rpolczer/sourcetree/projects/mwp2/.venv/bin/activate


python src/market_prediction_workbench/data.py
poetry run python src/market_prediction_workbench/train.py
python src/market_prediction_workbench/evaluate.py
