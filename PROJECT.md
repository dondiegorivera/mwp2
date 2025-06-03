Below is a “battle-plan” that takes you from the raw panel file you showed to a production-ready model that—if you execute all the way—*should* leave your friend’s 3-layer probabilistic MLP in the dust.
Every block is written so it can be handed straight to a coding team (or to yourself) and checked off one by one.

---

## 0 Project scaffold (1 day) ✅

| Folder         | Purpose                                                |
| -------------- | ------------------------------------------------------ |
| `data/`        | raw CSV drops and processed parquet files              |
| `notebooks/`   | ad-hoc EDA & sanity checks                             |
| `src/`         | reusable Python packages (⚠️ **no** notebooks in here) |
| `conf/`        | YAML configs (hyper-params, paths, experiment IDs)     |
| `experiments/` | Hydra or MLflow runs (auto-logged)                     |
| `tests/`       | pytest unit tests for every preprocessing & model util |

> **Tip:** Use `poetry` or `pip-tools` + `pre-commit` so deterministic environments are painless.

---

## 1 Data pipeline (2-3 days) ✅

1. **Ingest & type-cast**

   ```python
   df = pl.read_csv("data/raw/panel.csv", parse_dates=["Date"])  # polars is fast
   df = df.with_columns([
       pl.col("Closing Price").alias("close"),
       pl.col("Opening Price").alias("open"),
       pl.col("Volume").cast(pl.Int64)
   ])
Use code with caution.
Markdown
Symbol mapping → contiguous ticker_id integers.
Trading-calendar reindex
Per ticker → forward-fill weekends/holidays up to 5 days gap; create is_missing mask feature.
Target definition
Primary: next-day log-return
y_{t+1} = log(close_{t+1}) – log(close_t)
Sliding-window dataset builder (torch.utils.data.Dataset)
window length lookback = 120 days
prediction horizon h = 1, 5, 20 days (multi-output)
static features (broadcast once per sample):
ticker embedding id
sector one-hot / embedding
known-future features (calendar): day-of-week, month, end-of-quarter flag
observed-past features: log-returns, volume Z-score, 20-day realised vol, 14-day RSI, MACD, rolling skew/kurtosis
Unit-test: assert not batch['x'].isnan().any() after all transforms.
2 Model architecture — Temporal-Fusion-Transformer + ticker embedding (2 weeks) ✅
Why TFT?
Interpretable variable-selection & attention
Handles mixed static / past-observed / known-future covariates out of the box
Quantile & point forecasts simultaneously
Skeleton (PyTorch + pytorch-forecasting)
class GlobalTFT(pl.LightningModule):
    def __init__(self, conf): # conf would be model_specific_params, lr, etc.
        super().__init__()
        # self.save_hyperparameters(conf) # Better to save specific args
        # self.model = TFT.from_dataset(timeseries_dataset, ...)
        # ... constructor passes timeseries_dataset and parameters ...
        # Example from actual implementation:
        # self.model = TemporalFusionTransformer.from_dataset(
        #     timeseries_dataset,
        #     learning_rate=learning_rate, # TFT takes LR here
        #     **model_specific_params
        # )
        # ... training_step, validation_step, configure_optimizers ...
        pass # Refer to actual src/market_prediction_workbench/model.py
Use code with caution.
Python
Ticker embedding dim = min(round(num_tickers ** 0.25), 32)
Use WeightedSampler so each epoch sees the same #samples per ticker → stops AAPL dominating.
Loss: Pinball (quantile) averaged over the three horizons.
Done: TFT backbone + balanced sampler + quantile loss.
Artifacts: experiments/B2/*.
Next: Baseline & ablation (Step 3).
3 Baseline & ablation checklist (4–5 days)
ID	Variation	Hypothesis	Pass criteria
B0	Friend’s 3-layer MLP	sanity baseline	matches his RMSE
B1	TFT w/out ticker emb	global patterns only	↑ CV coverage, ↓ RMSE vs B0
B2	TFT + ticker emb	ID info helps	beats B1
B3	TFT + sector emb	sector shared α	beats B2
B4	PatchTST encoder (eff. transformer)	long memory helps	beats B3 on 20-day horizon
Automate with Hydra:
python train.py model=tft data=lookback120 horizon=multi exp_id=B3
Use code with caution.
Bash
4 Hyper-parameter sweep (2–3 days GPU time)
hidden_size: {32, 64, 128}
lr: {3e-4, 1e-3, 3e-3} with cosine decay
lookback: {60, 120, 250}
Optuna (pruned on ⅓ epoch) + WandB sweeps.
Objective: validation p50 MAE (1-day) + coverage@95% between 92-98%.
5 Ensemble for the kill (1 day)
Top-5 checkpoints (by val metric) → median-of-means ensemble.
Empirically gives ~3-8 % extra improvement & tighter interval calibration.
6 Walk-forward back-test & trading sim (3 days)
Roll window: train up to 2018, test 2019; slide yearly until 2025.
Benchmarks:
Random-walk (r̂ = 0)
Friend’s model
Simple 20-day momentum
Metrics: Sharpe, max-drawdown, hit-ratio, turnover.
Transaction cost assumption: 5 bp each side.
Position sizing: Kelly fraction using model σ.
Criteria to declare victory: Your ensemble >15 % annualised risk-adjusted return over friend’s model net of costs.
7 Productionisation (4–5 days)
Task	Deliverable
torch.jit.trace → model.pt	single binary
REST endpoint (FastAPI)	/predict?ticker=BMW.DE&date=2025-05-26
Batch scheduler	nightly retrain (incremental)
Monitoring	Data drift (Kolmogorov), coverage drift; send Slack alert if 95 % PI <80 % or >99 % for 3 straight days
8 Extra power-ups (stretch)
Mixture-of-Experts head
Add gating network on macro regime (VIX, rates ↑/↓).
Graph message-passing layer before TFT
Edge weights = rolling 60-day correlation; captures contagion.
Meta-learning fine-tune
MAML loop so a tiny 10-step gradient hop personalises to brand-new IPOs.
9 What to hand your dev team
README.md with setup + make all command
src/ package (data.py, model.py, train.py, evaluate.py)
conf/ defaults & Hydra job overrides
Unit tests covering every transform
A Jupyter notebook that reproduces the final benchmark table
Why this will beat the 3-layer MLP
Dimension	Friend’s MLP	Your TFT ensemble
Look-back memory	4 lags	120 (or 250) days
Covariates	just price lags	price, vol, tech indicators, calendar, sector
Architecture	linear-ish MLP	gated LSTM + attention + static context
Probabilistic	Normal only	pinball quantiles (any distribution)
Over-fit control	none	dropout, weight decay, bagging
Interpretability	none	attention heat-maps & variable-importance
Multi-horizon	no	yes (1 d, 5 d, 20 d)
Bigger receptive field + richer features + calibrated uncertainty ≈ strictly more information absorbed and exploited—so unless your friend secretly has perfect feature engineering, you win.
Now roll up your sleeves and start ticking those boxes. Happy model-hunting!
**7. `README.md` Updates**

```markdown
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

## Quick-start

To run a default TFT model training:
```bash
poetry run python -m market_prediction_workbench.train model=tft_default data=lookback120
Use code with caution.
To run a quick 1-epoch smoke test:
poetry run python src/market_prediction_workbench/train.py \
  model=tft_default \
  data=lookback120 \
  trainer.max_epochs=1
Use code with caution.
Bash
Project Structure
data/: Raw and processed data.
notebooks/: EDA and scratch work.
src/market_prediction_workbench/: Core Python package.
conf/: Configuration files.
experiments/: Experiment logs and outputs.
tests/: Unit tests.
These changes should fulfill all the requirements listed in your finish-line checklist. Remember to have the `stock_data.csv` in `data/raw/` and run the data processing pipeline (`python src/market_prediction_workbench/data.py`) before attempting to train, so that `data/processed/processed_data.parquet` is available. The tests also rely on this processed file.
