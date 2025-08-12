# Prompt — Lean step-by-step refactor plan from my plan + code

You are a senior software and ML engineer. I will provide:

* **PLAN**: my refactor intentions (text).
* **CODE**: small codebase (\~2k LOC) or key files.

Produce a **short, logically ordered instruction set** for implementing the refactor with minimal risk. Keep it practical and concise.

## Rules

* Do **not** invent files/APIs; only use what’s in CODE. If you must assume, prefix with `ASSUMPTION:` and proceed.
* No time estimates, no Jira terms, no fluff.
* Prefer **small, self-contained steps** that build safely on each other.
* Preserve current behavior and public APIs unless a step is explicitly marked **BREAKING**.
* Keep each step \~10–15 lines max.

## Output structure (strict)

1. **Overview (5–10 lines max)**

   * One-paragraph summary of the approach and key risks.
   * Any steps that can run **in parallel** (mark clearly), everything else is sequential.

2. **Steps (numbered, strictly ordered)**

```
Step [N]: [Short Title] [add **BREAKING** if applicable]
Why:
- One sentence: intent/benefit of this step.

Targets:
- Exact files/functions/classes to touch (paths relative to repo root).

Instructions:
- Precise changes (rename/extract/split/move/invert dependency/add adapter).
- Any signatures/config keys to add/change.
- Invariants to keep (behavior, API, performance boundary if relevant).

Tests:
- New/updated tests (brief cases or assertions).

Verify:
- Commands to run (format/lint/tests); quick runtime/manual check if needed.

Dependencies:
- Steps that must precede this (or “None”).
```

3. **Final checks (bullet list, ≤8 bullets)**

* Confirm key risks addressed, leftover follow-ups (if any), and simple rollback guidance.

## Quality bar

* Steps must be **runnable independently** and naturally map to single commits.
* Minimal surface area per step; avoid mixed concerns.
* Clear verification so I can tell if the step is done without guesswork.

## Inputs start here

PLAN:
\[PASTE YOUR PLAN]

CODE (tree + any key snippets):
\[PASTE REPO TREE + FILE EXCERPTS]

---

I will submit the plan and the codebase in 3 steps so it will fit into the input window. After the first 1st and the 2nd input, please only confirm that you received it. After the 3rd input you can start working on the plan.


INPUT 1: DETAILED PLAN
---

# 1) Data correctness & leakage (highest ROI)

**A. Split by date, not `time_idx`, and add a gap.**
Right now `time_idx` is re-ranked *per ticker*. Using `cutoff = int(max(time_idx)*0.8)` across the whole table mixes incomparable indices. Do:

* Global **date-based** split with an **exclusion gap** of `lookback + max_horizon` trading days (or ≥30 calendar days).
* Better: **purged, embargoed walk-forward** (De Prado style) for all CV and final eval.

**B. Use a proper trading calendar, not daily upsample.**
Weekend gaps are fine, but exchange holidays and half-days aren’t. Forward-filling over a 4-day long weekend (or different markets) skews volatility estimates. Adopt an exchange calendar (e.g., NYSE) and **reindex to trading days**; if you must remain daily, tag non-trading days and **do not** roll statistics across them.

**C. Minimize forward-fill leakage.**
You forward-fill many features then compute rolling stats. You masked `is_missing` in places (good), but ensure **rolling stats never span missing bars**. If a bar is synthetic, drop it from rolling windows; don’t let it propagate “quiet” days that never traded.

**D. Outlier “drop” on targets is risky.**
Clipping by dropping (`MAX_ABS_DAILY_RETURN`) removes exactly the moves you most care about for P\&L. Prefer **winsorization** (clip values) or a **robust loss** (Huber/quantile) rather than deleting rows (which biases conditioning and can break sequence continuity).

**E. Survivorship & delisting bias.**
Make sure your raw panel includes delisted names or, at minimum, add a **“live” mask** per ticker to avoid training on “only survivors.” If not available, restrict to an index membership snapshot **valid at each date**.

---

# 2) Feature quality (second highest ROI)

You partially solved the “data silo” (added market return), but there’s more juice:

**A. Calendar features: make them cyclical; don’t standardize booleans.**

* Replace `day_of_week`, `month` with **sin/cos** encodings.
* Keep `is_quarter_end` as 0/1 and **do not scale** it (currently GroupNormalizer can distort 0/1 flags). Exclude calendar dummies from group normalizers.

**B. Cross-sectional & relative features (alpha bread-and-butter).**

* **Residual (idiosyncratic) return**: `r_ticker − β⋅r_market` where `β` is rolling (e.g., 60d).
* **Sector‐neutral return**: `r_ticker − mean_sector_return`.
* **Rank features** (per day): rank of 1d/5d momentum, volume z-score, RSI—ranks are robust and reduce scale drift.
* **Overnight vs intraday**: `log(open_t/close_{t-1})`, `log(close_t/open_t)`; different alpha.
* **Liquidity proxy**: ADV(20), turnover (vol/market\_cap), and their z-scores.
* **Volatility regime**: rolling σ and its z-score; optionally **realized downside vol**.

**C. Broader context (macro/regime):**

* **VIX**, 10Y yield, 2s10s spread, DXY, commodities index; encode as **known-future (slow)** or **unknown (fast)** appropriately.
* **Market state** bucket (low/med/high vol) as a categorical for a small gating head or as a feature.

**D. Sentiment / events (optional but strong):**

* Headline-level **FinBERT** sentiment and **news volume** per ticker/day, or simpler **earnings day indicator** and `days_since_last_earnings`. Even these two tiny features move the needle.

**E. Target design sanity:**

* Predict **future log-returns** is fine. Consider **direction classification** as an auxiliary head (see §4) and keep **multi-horizon** but be sure evaluation truly uses all horizons.

---

# 3) Normalization & encoders (stability fixes)

**A. Do not GroupNormalize calendar/boolean features.**
GroupNormalizer per ticker for returns/vol/volume is good. Keep **binary flags as float 0/1 unscaled**, and keep **cyclical sin/cos** raw or globally standardized (not per ticker).

**B. Target normalizer choice.**
You correctly switched away from GroupNormalizer for returns. **EncoderNormalizer** is acceptable; also test “**identity**” (no scaling) with quantile loss (returns are already stationary).

**C. Be explicit on feature groups** in the TimeSeriesDataSet config—double-check every column lands in the intended bucket. PF is powerful but strict; mis-bucketing known/unknown reals can silently leak or starve signal.

---

# 4) Losses, heads, and what you actually optimize

Right now your default config sets **DirectionalLoss** (point forecast) while **evaluation assumes quantiles** (coverage\@90). That guarantees nonsense coverage and weak gradients for intervals.

**A. Separate objectives cleanly:**

* **Primary head**: **Quantile loss** for predictive distribution (p05,p50,p95).
* **Auxiliary head**: **Direction classification** (BCE on sign of 1d/5d) or your DirectionalLoss as **aux loss**. Weight it (e.g., 0.7 quantile, 0.3 direction). Your `hybrid.yaml` points exactly here—use it as default.
* Optional: **Expectile loss** (asymmetric MSE) is a great middle ground for tails.

**B. Horizon handling correctness.**
Your `evaluate.py` collapses to `preds_h1 = preds_dec[:, 0, :, :]` and then loops targets; that almost certainly mismaps **horizon vs target** axes. Make sure the “T” dimension in `[B, H, T, Q]` is truly targets vs horizons. Many PF pitfalls come from this single mistake. Verify: **H = prediction length (steps ahead)**, **T = #targets**. Then compute metrics **per horizon** (1d, 5d, 20d), not just horizon 0.

**C. Calibration.**
Post-train, apply **quantile calibration** (simple affine/beta scaling over validation) and/or **conformal** adjustment to hit nominal 90–95% coverage across time and tickers. Poor calibration → poor risk sizing.

---

# 5) Sampling, batching, and CV

**A. Reinstate balanced sampling.**
You removed `WeightedRandomSampler` in train. Bring it back (or per-batch uniform over tickers). Without it, mega-cap histories dominate.

**B. Proper CV.**
Replace single split with **expanding walk-forward** or **rolling windows** (e.g., yearly steps 2014→2018 train, 2019 test …). Aggregate metrics with **purging** to avoid lookahead via overlapping windows.

**C. Batch size realism.**
`batch_size: 4096` for TFT can be too big or too small depending on GPU/seq length. Favor **grad accumulation** + **moderate batches** (e.g., 256–1024) to keep temporal dynamics stable. Watch validation loss jitter.

---

# 6) Evaluation that maps to trading (you need this to know if it “works”)

Your current metrics (MAE, RMSE, coverage) are **necessary but not sufficient**. Add a simple but honest backtest:

**A. Signals:**

* **Direction** (sign of p50) and **magnitude** (p50 itself) with uncertainty (p05/p95).
* Build a **position scaler** `w_t = clip(p50 / predicted_sigma, -w_max, w_max)` (vol targeting).

**B. Strategy templates:**

* **Long-only**: go long if p50 > threshold, cash otherwise.
* **Long/short**: rank by p50/σ, take top/bottom deciles with dollar neutral and **turnover cap**.
* Enforce **costs** (e.g., 5–10 bps each way), **borrow penalty** on shorts, and **slippage**.

**C. Metrics:**

* **Sharpe, Sortino, MaxDD, Calmar, Hit-rate, Turnover, Capacity** (notional scaled by ADV).
* **Information Coefficient (Spearman)** of predicted vs realized return per day (cross-sectional alpha health).

**D. Purge & embargo in backtest** consistent with the training split to avoid look-ahead.

This will quickly tell you whether gains come from a tiny subset, overfitting, or are broadly persistent.

---

# 7) Architecture: TFT is fine—just make it “finance-aware”

If you still struggle after fixes above:

**A. Try a **PatchTST** encoder** (superb for long memory) as a drop-in baseline on the same features. It often beats TFT for pure numeric panels.

**B. Small **graph layer** before TFT.**
Daily 60-day rolling correlation graph → message passing (2–3 layers, tiny hidden). This captures contagion and sector moves cheaply.

**C. Regime gating.**
A micro **mixture-of-experts** where a 3-class regime head (low/med/high vol) gates output heads tends to help drawdowns.

Keep it small—don’t out-parameterize the signal.

---

# 8) Training hygiene & robustness

* **Learning rate schedule**: cosine anneal with warmup or OneCycle often stabilizes TFT more than ReduceLROnPlateau.
* **Dropout/weight decay**: modest (0.1 / 1e-5 OK). Add **variational dropout** on LSTM layers if overfitting.
* **Mixed precision** + **grad clipping** (already on).
* **Seeded W\&B sweeps** on: `lookback {60, 120, 252}`, `hidden {64, 128}`, `dropout {0.1, 0.2}`, `aux weight {0.2, 0.4, 0.6}`.
* **Early stopping**: monitor **per-horizon** validation (1d often noisy; 5d more stable).
* **Unit tests**: add test that **known-future** columns are equal across encoder/decoder and **unknown** are masked properly at prediction time.

---

# 9) Interpretability that prevents self-delusion

Use TFT’s variable selection and attention—but *sanity check*:

* If **market\_return\_1d** or **volatility regime** isn’t top-3 important globally, feature engineering is off.
* On case studies (earnings weeks), attention should weight **recent** bars and **volatility** features. If you see attention on long-ago points with no narrative, that’s often data alignment or scaling error.

---

# 10) Productionization (after the edge is real)

* **Model registry**: save **data schema hash**, feature list, normalizer params, and the **training date window** with the checkpoint.
* **Data drift monitors**: PSI/KS on per-feature *by ticker* and globally; **coverage drift** alarms.
* **Shadow backtest** live: compute daily IC & realized PnL of the current model against a holdout universe before risking capital.

---

## Quick “must-fix” checklist (shortlist)

1. Switch to **date-based, purged walk-forward** splits.
2. **Trading calendar** reindex; avoid rolling stats across synthetic bars.
3. **Stop dropping** target outliers; clip or robust loss instead.
4. **Don’t normalize** booleans; **cyclical encode** calendar.
5. Reinstate **balanced sampler**.
6. Make **quantile loss primary**, **directional/classification auxiliary**; fix evaluation tensor axes so horizons are evaluated correctly.
7. Add **backtest with costs**, position sizing by predicted uncertainty, and report **Sharpe/MaxDD/Turnover/IC**.
8. Add **cross-sectional & residual features** (beta, sector-neutral, ranks, overnight).
9. Add **macro/regime** features (VIX, yields) and a tiny **regime gate**.
10. **Quantile calibration / conformal** for interval reliability.

Do those, and you’ll go from “statistically okay but not tradable” → “honest signal with risk-aware sizing.” From there, iterate features and horizon mix while ruthlessly checking out-of-sample Sharpe under costs. That’s the path to a model you can actually put money behind.

---
INPUT 2: FIRST HALF OF THE EXISTING CODE
---

===== Project File Tree =====
├── .gitignore
├── .pre-commit-config.yaml
├── .ruff_cache
├── .venv
├── GPT5_Plan.md
├── README.md
├── bak
├── conf
│   ├── config.yaml
│   ├── data
│   ├── model
│   │   └── tft_default.yaml
│   └── trainer
│       └── default_trainer.yaml
├── data
├── experiments
├── lightning_logs
├── market-prediction-workbench
│   └── data
├── notebooks
├── outputs
├── poetry.lock
├── pyproject.toml
├── resources
├── src
│   ├── __init__.py
│   └── market_prediction_workbench
│       ├── __init__.py
│       ├── __pycache__
│       ├── data.py
│       ├── evaluate.py
│       ├── model.py
│       └── train.py
└── tests
    ├── __init__.py
    ├── test_data.py
    └── test_model.py

===== Code and Configuration Files =====

===== .pre-commit-config.yaml =====
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
        enabled: false  # <-- disables black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
        enabled: false  # <-- disables ruff-format

===== src/__init__.py =====


===== src/market_prediction_workbench/__init__.py =====


===== src/market_prediction_workbench/train.py =====
# src/market_prediction_workbench/train.py
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl
import polars as pl_df  # Renamed to avoid conflict with pytorch_lightning.pl
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import shutil
from hydra.core.hydra_config import HydraConfig

# Import our custom modules
from market_prediction_workbench.model import GlobalTFT

# Import pytorch-forecasting specific items
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import (
    GroupNormalizer,
    NaNLabelEncoder,
    EncoderNormalizer,
    MultiNormalizer,
)

# Import Lightning Callbacks
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

# Import scikit-learn's StandardScaler if we intend to use it
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

if torch.cuda.is_available():
    try:
        if torch.cuda.get_device_capability()[0] >= 7:
            torch.set_float32_matmul_precision("medium")
            print("PyTorch float32 matmul precision set to 'medium' for Tensor Cores.")
        else:
            print(
                "Current GPU does not have Tensor Cores (or CC < 7.0). Default matmul precision used."
            )
    except Exception as e:
        print(
            f"Could not set matmul precision (may be normal if no CUDA GPU or older PyTorch): {e}"
        )
else:
    print("CUDA not available. Running on CPU. Matmul precision setting skipped.")


# RESTORED: Full, verbose, and robust helper function from your original file
def get_embedding_sizes_for_tft(timeseries_dataset: TimeSeriesDataSet) -> dict:
    embedding_sizes = {}

    # Access the internal attribute _categorical_encoders, which is populated by TimeSeriesDataSet
    dataset_encoders = timeseries_dataset._categorical_encoders

    # Check if _categorical_encoders attribute exists and is a non-empty dictionary
    if not isinstance(dataset_encoders, dict) or not dataset_encoders:
        if (
            timeseries_dataset.categoricals
        ):  # Check if there are any categoricals defined in the dataset
            print(
                "Warning (get_embedding_sizes_for_tft): TimeSeriesDataSet._categorical_encoders is missing, not a dict, or empty, "
                "but dataset has categorical columns. TFT might use defaults or error."
            )
        return {}  # Return empty if no encoders or not a dict

    print(
        f"DEBUG (get_embedding_sizes_for_tft): Processing encoders from TimeSeriesDataSet._categorical_encoders: {dataset_encoders}"
    )

    for col_name in (
        timeseries_dataset.categoricals
    ):  # Iterate over actual categoricals defined in dataset
        if col_name in dataset_encoders:
            encoder = dataset_encoders[col_name]
            print(
                f"DEBUG (get_embedding_sizes_for_tft): Encoder for '{col_name}': {encoder}, type: {type(encoder)}"
            )

            cardinality_val = None
            # Try .cardinality property first (NaNLabelEncoder has this)
            if hasattr(encoder, "cardinality"):
                try:
                    cardinality_val = encoder.cardinality
                    if cardinality_val is not None:
                        print(
                            f"DEBUG (get_embedding_sizes_for_tft): Accessed encoder.cardinality for '{col_name}': {cardinality_val}"
                        )
                    else:
                        print(
                            f"DEBUG (get_embedding_sizes_for_tft): encoder.cardinality for '{col_name}' returned None."
                        )
                except AttributeError:
                    print(
                        f"DEBUG (get_embedding_sizes_for_tft): AttributeError on encoder.cardinality for '{col_name}'. Will try .classes_."
                    )
                    cardinality_val = None

            if cardinality_val is None:  # Fallback or if .cardinality was None
                if hasattr(encoder, "classes_") and encoder.classes_ is not None:
                    num_classes = len(encoder.classes_)
                    add_nan_flag = False
                    # Check for add_nan attribute (specific to NaNLabelEncoder but good general check)
                    # NaNLabelEncoder is the main one that uses add_nan and contributes to cardinality this way
                    if hasattr(encoder, "add_nan") and isinstance(
                        encoder, NaNLabelEncoder
                    ):
                        add_nan_flag = encoder.add_nan

                    cardinality_val = num_classes + (1 if add_nan_flag else 0)

                    print(
                        f"DEBUG (get_embedding_sizes_for_tft): Calculated cardinality from len(encoder.classes_) for '{col_name}': {cardinality_val} (add_nan={add_nan_flag})"
                    )
                else:
                    print(
                        f"ERROR (get_embedding_sizes_for_tft): Could not determine cardinality for '{col_name}'. Skipping."
                    )
                    continue

            if cardinality_val is None:  # Should not happen if logic above is complete
                print(
                    f"ERROR (get_embedding_sizes_for_tft): Cardinality for '{col_name}' is unexpectedly None. Skipping."
                )
                continue

            # For TFT, cardinality must be at least 1.
            tft_cardinality = max(1, cardinality_val)

            # Calculate embedding dimension
            if tft_cardinality <= 1:  # e.g. only one unique value or only NaNs
                dim = 1
            else:
                # Using the project's original formula
                dim = min(round(tft_cardinality**0.25), 32)
                dim = max(1, int(dim))  # Ensure dim is at least 1

            embedding_sizes[col_name] = (tft_cardinality, dim)
            print(
                f"DEBUG (get_embedding_sizes_for_tft): Setting embedding for '{col_name}': ({tft_cardinality}, {dim})"
            )
        else:
            print(
                f"Warning (get_embedding_sizes_for_tft): Categorical column '{col_name}' (from dataset.categoricals) "
                f"not found in TimeSeriesDataSet._categorical_encoders. This is unexpected if encoders were meant to be created for all."
            )

    if not embedding_sizes and timeseries_dataset.categoricals:
        print(
            "Warning (get_embedding_sizes_for_tft): Resulting embedding_sizes dictionary is empty, but dataset has categoricals. TFT will use defaults or error."
        )
    elif embedding_sizes:
        print(f"Calculated embedding_sizes for TFT: {embedding_sizes}")
    return embedding_sizes


def split_before(ds: TimeSeriesDataSet, pct: float = 0.8):
    cutoff = int(ds.data[ds.time_idx].max() * pct)
    train_df = ds.data[ds.data[ds.time_idx] <= cutoff]
    val_df = ds.data[ds.data[ds.time_idx] > cutoff]
    return (
        TimeSeriesDataSet.from_dataset(ds, train_df),
        TimeSeriesDataSet.from_dataset(ds, val_df),
    )


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    pl.seed_everything(cfg.seed, workers=True)

    processed_data_path = Path(cfg.paths.processed_data_file)
    if not processed_data_path.exists():
        print(f"Processed data not found at {processed_data_path}.")
        print(
            "Please run the data processing pipeline first (e.g., python src/market_prediction_workbench/data.py)"
        )
        return

    polars_data_df = pl_df.read_parquet(processed_data_path)
    data_pd = polars_data_df.to_pandas()
    print(f"Loaded and converted to Pandas DataFrame. Shape: {data_pd.shape}")

    time_idx_col_name = str(cfg.data.time_idx)
    if time_idx_col_name in data_pd.columns:
        data_pd[time_idx_col_name] = data_pd[time_idx_col_name].astype(np.int64)
    else:
        raise ValueError(
            f"Time index column '{time_idx_col_name}' not found for casting."
        )

    def get_list_from_cfg_node(config_node_val):
        if config_node_val is None:
            return []
        if isinstance(config_node_val, (str, int, float)):
            return [str(config_node_val)]
        if isinstance(config_node_val, (list, ListConfig)):
            return [str(item) for item in config_node_val]
        raise TypeError(
            f"Expected list or primitive for config node, got {type(config_node_val)}"
        )

    group_ids_list = get_list_from_cfg_node(
        OmegaConf.select(cfg, "data.group_ids", default=[])
    )
    target_list = get_list_from_cfg_node(
        OmegaConf.select(cfg, "data.target", default=[])
    )
    static_categoricals_list = get_list_from_cfg_node(
        OmegaConf.select(cfg, "data.static_categoricals", default=[])
    )
    static_reals_list = get_list_from_cfg_node(
        OmegaConf.select(cfg, "data.static_reals", default=[])
    )
    time_varying_known_categoricals_list = get_list_from_cfg_node(
        OmegaConf.select(cfg, "data.time_varying_known_categoricals", default=[])
    )
    time_varying_known_reals_list = get_list_from_cfg_node(
        OmegaConf.select(cfg, "data.time_varying_known_reals", default=[])
    )
    time_varying_unknown_categoricals_list = get_list_from_cfg_node(
        OmegaConf.select(cfg, "data.time_varying_unknown_categoricals", default=[])
    )
    time_varying_unknown_reals_list = get_list_from_cfg_node(
        OmegaConf.select(cfg, "data.time_varying_unknown_reals", default=[])
    )
    time_idx_str = str(cfg.data.time_idx)

    all_categorical_cols_from_config = list(
        dict.fromkeys(
            static_categoricals_list
            + time_varying_known_categoricals_list
            + time_varying_unknown_categoricals_list
        )
    )
    for cat_col_name_str in all_categorical_cols_from_config:
        if cat_col_name_str in data_pd.columns:
            if (
                data_pd[cat_col_name_str].dtype != object
                and data_pd[cat_col_name_str].dtype != str
                and not pd.api.types.is_string_dtype(data_pd[cat_col_name_str])
            ):
                print(
                    f"Casting categorical column '{cat_col_name_str}' to string. Original dtype: {data_pd[cat_col_name_str].dtype}"
                )
                data_pd[cat_col_name_str] = data_pd[cat_col_name_str].astype(str)
        else:
            print(
                f"Warning: Configured categorical column '{cat_col_name_str}' not found in DataFrame for dtype casting."
            )

    # --- CORRECTED: Split DataFrame BEFORE creating TimeSeriesDataSet objects ---
    max_time_idx = data_pd[time_idx_str].max()
    train_cutoff_idx = int(max_time_idx * 0.8)
    print(f"Splitting data for training/validation at time_idx: {train_cutoff_idx}")

    train_df = data_pd[data_pd[time_idx_str] <= train_cutoff_idx]
    val_df = data_pd[data_pd[time_idx_str] > train_cutoff_idx]

    print(f"Training DataFrame shape: {train_df.shape}")
    print(f"Validation DataFrame shape: {val_df.shape}")

    # Common parameters for both datasets
    dataset_params = dict(
        time_idx=time_idx_str,
        target=target_list[0] if len(target_list) == 1 else target_list,
        group_ids=group_ids_list,
        max_encoder_length=cfg.data.lookback_days,
        max_prediction_length=cfg.data.max_prediction_horizon,
        static_categoricals=static_categoricals_list,
        static_reals=static_reals_list,
        time_varying_known_categoricals=time_varying_known_categoricals_list,
        time_varying_known_reals=time_varying_known_reals_list,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals_list,
        time_varying_unknown_reals=time_varying_unknown_reals_list,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # Scaler and Normalizer Logic (applied to training_dataset first)
    scalers = {}
    if cfg.data.get("scalers") and cfg.data.scalers.get("default_reals_normalizer"):
        default_normalizer_name = cfg.data.scalers.default_reals_normalizer
        reals_to_scale = list(
            dict.fromkeys(
                time_varying_unknown_reals_list
                + time_varying_known_reals_list
                + static_reals_list
            )
        )
        for col in reals_to_scale:
            if default_normalizer_name == "GroupNormalizer":
                scalers[col] = GroupNormalizer(groups=group_ids_list, method="standard")
            elif default_normalizer_name == "EncoderNormalizer":
                scalers[col] = EncoderNormalizer()
            elif default_normalizer_name == "StandardScaler":
                scalers[col] = SklearnStandardScaler()
    else:
        print(
            "No 'default_reals_normalizer' specified. Using GroupNormalizer as default."
        )
        reals_to_scale = list(
            dict.fromkeys(
                time_varying_unknown_reals_list
                + time_varying_known_reals_list
                + static_reals_list
            )
        )
        for col in reals_to_scale:
            scalers[str(col)] = GroupNormalizer(
                groups=group_ids_list, method="standard"
            )

    single_target_normalizer_prototype_name = OmegaConf.select(
        cfg, "data.scalers.target_normalizer", default="GroupNormalizer"
    )
    final_target_normalizer = None
    if single_target_normalizer_prototype_name:
        if len(target_list) > 1:
            normalizers_list = []
            for _ in target_list:
                if single_target_normalizer_prototype_name == "GroupNormalizer":
                    normalizers_list.append(
                        GroupNormalizer(
                            groups=group_ids_list,
                            method="standard",
                        )
                    )
                elif single_target_normalizer_prototype_name == "EncoderNormalizer":
                    normalizers_list.append(EncoderNormalizer())
                elif single_target_normalizer_prototype_name == "StandardScaler":
                    normalizers_list.append(SklearnStandardScaler())
            if normalizers_list:
                final_target_normalizer = MultiNormalizer(normalizers=normalizers_list)
        elif target_list:
            if single_target_normalizer_prototype_name == "GroupNormalizer":
                final_target_normalizer = GroupNormalizer(
                    groups=group_ids_list, method="standard"
                )
            elif single_target_normalizer_prototype_name == "EncoderNormalizer":
                final_target_normalizer = EncoderNormalizer()
            elif single_target_normalizer_prototype_name == "StandardScaler":
                final_target_normalizer = SklearnStandardScaler()

    # Create training dataset. This dataset "learns" the scalers.
    print("Creating training TimeSeriesDataSet...")
    training_dataset = TimeSeriesDataSet(
        train_df,
        **dataset_params,
        scalers=scalers,
        target_normalizer=final_target_normalizer,
    )
    print("Training TimeSeriesDataSet created successfully.")

    # Create validation dataset from the training dataset to ensure scalers are reused.
    print("Creating validation TimeSeriesDataSet from training dataset...")
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, val_df, allow_missing_timesteps=True
    )
    print("Validation TimeSeriesDataSet created successfully.")

    if len(training_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError(
            "Train/validation split resulted in an empty dataset. Check split logic and data range."
        )
    print(
        f"Training samples: {len(training_dataset)}, Validation samples: {len(validation_dataset)}"
    )

    calculated_embedding_sizes = get_embedding_sizes_for_tft(training_dataset)

    model_module = hydra.utils.get_class(cfg.model._target_)
    model_specific_params_from_cfg = {
        k: v
        for k, v in cfg.model.items()
        if k not in ["_target_", "learning_rate", "weight_decay"]
    }

    if "loss" in model_specific_params_from_cfg and isinstance(
        model_specific_params_from_cfg["loss"], DictConfig
    ):
        model_specific_params_from_cfg["loss"] = hydra.utils.instantiate(
            model_specific_params_from_cfg["loss"]
        )

    model_specific_params_from_cfg["embedding_sizes"] = calculated_embedding_sizes

    model = model_module(
        timeseries_dataset=training_dataset,  # Initialize model with training dataset parameters
        model_specific_params=model_specific_params_from_cfg,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )
    print(f"Model {cfg.model._target_} (GlobalTFT wrapper) initialized.")

    # REMOVED WeightedRandomSampler logic
    print("Using default shuffling for train_loader.")
    num_cpu = os.cpu_count()
    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=cfg.trainer.batch_size,
        num_workers=(
            min(num_cpu - 2, cfg.trainer.num_workers)
            if num_cpu and num_cpu > 2
            else cfg.trainer.num_workers
        ),
        shuffle=True,
        pin_memory=True,
        persistent_workers=True if cfg.trainer.num_workers > 0 else False,
        prefetch_factor=4,
    )

    val_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=cfg.trainer.batch_size * 2,
        num_workers=cfg.trainer.num_workers,
        shuffle=False,
        drop_last=False,
    )

    early_stop_callback = EarlyStopping(
        monitor=cfg.trainer.early_stopping_monitor,
        patience=cfg.trainer.early_stopping_patience,
        mode=cfg.trainer.early_stopping_mode,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(
        logging_interval=cfg.trainer.lr_monitor_logging_interval
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        filename="{epoch}-{val_loss:.2f}-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    callbacks = [early_stop_callback, lr_monitor, checkpoint_callback]

    logger = None
    if cfg.trainer.get("use_wandb", False):
        from pytorch_lightning.loggers import WandbLogger

        run_name_wandb = f"{cfg.project_name}_{cfg.experiment_id}"
        logger = WandbLogger(
            name=run_name_wandb,
            project=cfg.trainer.wandb_project_name,
            entity=cfg.trainer.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            save_dir=str(Path(cfg.paths.log_dir) / "wandb"),
        )
        print("WandB Logger initialized.")
        if logger.log_dir:
            wandb_run_dir = Path(logger.log_dir)
            hydra_cfg_path = Path(HydraConfig.get().runtime.output_dir) / ".hydra"
            target_hydra_path = wandb_run_dir / ".hydra"
            print(
                f"Copying Hydra config from {hydra_cfg_path} to {target_hydra_path}..."
            )
            try:
                if target_hydra_path.exists():
                    shutil.rmtree(target_hydra_path)
                shutil.copytree(hydra_cfg_path, target_hydra_path)
                print("Successfully copied .hydra config directory.")
            except Exception as e:
                print(f"Error copying .hydra directory: {e}")
        else:
            print("Warning: Could not determine logger.log_dir. Skipping config copy.")
    else:
        print("WandB Logger is disabled.")

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=str(cfg.trainer.accelerator),
        devices=(
            cfg.trainer.devices
            if str(cfg.trainer.devices).lower() != "auto"
            else "auto"
        ),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 0.1),
        num_sanity_val_steps=0,
    )

    print("Trainer initialized. Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training finished.")


if __name__ == "__main__":
    main()


===== src/market_prediction_workbench/data.py =====
# src/market_prediction_workbench/data.py

import polars as pl
from pathlib import Path


# Add these imports at the top of src/market_prediction_workbench/data.py
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass


# --- Define a dataclass for configuration ---
@dataclass
class DataConfig:
    # Feature groups for the model
    static_reals: list[str]
    static_categoricals: list[str]
    time_varying_known_reals: list[str]
    time_varying_unknown_reals: list[str]
    target_columns: list[str]

    # Dataset parameters
    lookback_days: int = 120
    prediction_horizon: int = 1  # Not used for multi-horizon, but good practice


class MarketDataset(Dataset):
    def __init__(self, data: pl.DataFrame, config: DataConfig):
        super().__init__()
        self.config = config

        # Add an original index column to the data before any filtering for valid_indices
        # This index will be preserved through joins and filters.
        # Using with_row_index ensures we get an index from the current state of 'data'
        self.data = data.with_row_index(name="_original_idx_for_dataset")

        # Pre-calculate valid indices
        group_sizes = (
            self.data.group_by("ticker_id").len().rename({"len": "size"})
        )  # Use .len() as per warning

        data_with_size_and_orig_idx = self.data.join(group_sizes, on="ticker_id")

        max_horizon = 20  # Max prediction horizon

        # Filter to find rows that are valid end-points for a lookback window
        valid_rows_df = data_with_size_and_orig_idx.filter(
            (pl.col("time_idx") >= (config.lookback_days - 1))
            & (
                pl.col("time_idx") < (pl.col("size") - max_horizon)
            )  # Ensure targets up to max_horizon are available
        )

        # Select the original index column of these valid rows
        self.valid_indices = (
            valid_rows_df.select("_original_idx_for_dataset").to_series().to_list()
        )

        print(f"Created dataset with {len(self.valid_indices)} valid samples.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 1. Get the original DataFrame index for the end of the sample window
        original_df_idx = self.valid_indices[
            idx
        ]  # This is now a direct index into self.data

        # 2. Get the ticker and end time_idx for this sample
        #    end_row is the last row of the lookback window in the original DataFrame.
        end_row_data = self.data[
            original_df_idx
        ]  # Slice self.data using the direct index

        ticker_id = end_row_data.select("ticker_id").item()
        end_time_idx_for_sample = end_row_data.select(
            "time_idx"
        ).item()  # This is the time_idx of the last day in the lookback

        # 3. Slice the lookback window from self.data
        # We need to find rows in self.data that belong to this ticker_id
        # and fall within the time_idx range for the lookback window.
        # The lookback window ends at end_time_idx_for_sample.
        start_time_idx_for_sample = (
            end_time_idx_for_sample - self.config.lookback_days + 1
        )

        window_df = self.data.filter(
            (pl.col("ticker_id") == ticker_id)
            & (pl.col("time_idx") >= start_time_idx_for_sample)
            & (pl.col("time_idx") <= end_time_idx_for_sample)
        )

        # Ensure window_df has the correct length; if not, something is wrong with time_idx or data
        if window_df.height != self.config.lookback_days:
            # This can happen if data is very sparse even after ffill, or time_idx is not contiguous for a ticker
            # For robustness, one might pad or raise an error.
            # For now, let's raise an error to highlight if this occurs.
            raise ValueError(
                f"Sample {idx} for ticker {ticker_id} (orig_idx {original_df_idx}): "
                f"Expected lookback window of {self.config.lookback_days} days, "
                f"but got {window_df.height} days. "
                f"Time_idx range: {start_time_idx_for_sample} to {end_time_idx_for_sample}."
            )

        # 4. Target is on the end_row_data (which corresponds to time t for prediction at t+h)
        target_values_series = end_row_data.select(self.config.target_columns).row(
            0
        )  # Get as tuple

        # 5. Extract feature groups and convert to tensors
        # Static categoricals are taken from the first row of the window (they should be constant)
        x_static_cats_df = window_df.head(1).select(self.config.static_categoricals)
        x_static_cats = torch.tensor(
            x_static_cats_df.to_numpy(), dtype=torch.long
        ).squeeze(0)

        x_known_reals = torch.tensor(
            window_df.select(self.config.time_varying_known_reals).to_numpy(),
            dtype=torch.float32,
        )

        x_unknown_reals = torch.tensor(
            window_df.select(self.config.time_varying_unknown_reals).to_numpy(),
            dtype=torch.float32,
        )

        y = torch.tensor(
            target_values_series, dtype=torch.float32
        )  # Already a 1D sequence

        return {
            "x_cat": x_static_cats,  # Should be [num_static_categoricals]
            "x_known_reals": x_known_reals,  # Should be [lookback, num_known_reals]
            "x_unknown_reals": x_unknown_reals,  # Should be [lookback, num_unknown_reals]
            "y": y,  # Should be [num_targets]
            "groups": torch.tensor(
                [ticker_id], dtype=torch.long
            ),  # For pytorch-forecasting TimeSeriesDataSet compatibility
            "time_idx_window": torch.tensor(
                window_df.select("time_idx").to_series().to_list(), dtype=torch.long
            ),  # actual time_idx values in window
        }


# --- Helper function to clean column names ---
def _clean_col_names(df: pl.DataFrame) -> pl.DataFrame:
    """Converts all column names to snake_case."""
    return df.rename({col: col.lower().replace(" ", "_") for col in df.columns})


def load_and_clean_data(csv_path: Path) -> pl.DataFrame:
    """
    Loads the raw stock data from a CSV file, cleans column names,
    and performs initial type casting and column selection.
    """
    print(f"Loading data from {csv_path}...")

    # Define common null value representations
    common_null_values = ["N/A", "NA", "NaN", "", "null", "#N/A"]  # Added common ones

    # 1. Ingest & type-cast
    df = pl.read_csv(
        csv_path,
        try_parse_dates=True,
        null_values=common_null_values,  # <--- ADD THIS ARGUMENT
        infer_schema_length=10000,  # Recommended by Polars for large files
    )
    df = _clean_col_names(df)

    # 2. Select and rename core columns
    # The plan specifies 'close', 'open', 'volume'. We'll also keep ticker and date.
    # And the new 'market_cap' since it caused the error and might be useful later.
    # We will cast 'market_cap' to float, and Polars will handle N/A -> null conversion.
    df = df.select(
        pl.col("date"),
        pl.col("asset_ticker").alias("ticker"),
        pl.col("closing_price").alias("close"),
        pl.col("opening_price").alias("open"),
        pl.col("volume").cast(pl.Int64, strict=False),
        pl.col("industry").fill_null("N/A"),  # Fill missing industries
        pl.col("market_cap").cast(
            pl.Float64, strict=False
        ),  # Add market_cap, let errors turn to null
    )

    # Drop rows with nulls in essential columns (close, ticker, date)
    # Note: We are NOT dropping nulls for market_cap here, as it might be sparse.
    df = df.drop_nulls(subset=["date", "ticker", "close"])

    print("Initial cleaning and type casting complete.")
    print("DataFrame shape:", df.shape)
    print("Sample data:")
    print(df.head())

    return df


# Add this function to src/market_prediction_workbench/data.py


def create_mappings(df: pl.DataFrame, output_dir: Path) -> pl.DataFrame:
    """
    Creates integer mappings for 'ticker' and 'industry' and joins them back.
    Saves the mappings to parquet files for later use.
    """
    print("Creating ticker and industry mappings...")

    # Ticker mapping
    ticker_map = (
        df.select("ticker").unique().sort("ticker").with_row_index(name="ticker_id")
    )  # Use with_row_index

    # Industry mapping
    # Also sort 'industry' for consistent mapping if the order changes for some reason
    industry_map = (
        df.select("industry").unique().sort("industry").with_row_index(name="sector_id")
    )  # Use with_row_index

    # Save mappings
    output_dir.mkdir(parents=True, exist_ok=True)
    ticker_map.write_parquet(output_dir / "ticker_map.parquet")
    industry_map.write_parquet(output_dir / "industry_map.parquet")

    print(f"Saved mappings to {output_dir}")

    # Join back to the main dataframe
    df = df.join(ticker_map, on="ticker")
    df = df.join(industry_map, on="industry")

    return df


# Add this function to src/market_prediction_workbench/data.py

# src/market_prediction_workbench/data.py


def reindex_and_fill_gaps(df: pl.DataFrame, max_ffill_days: int = 5) -> pl.DataFrame:
    """
    Performs trading-calendar reindexing for each ticker.
    - Upsamples to a daily frequency.
    - Creates 'is_missing' mask.
    - Forward-fills data up to a specified limit.
    """
    print("Reindexing to a full calendar and forward-filling gaps...")

    # Ensure the dataframe is sorted by the grouping key and the time column
    # for the upsample operation to work correctly per group.
    df = df.sort("ticker_id", "date")

    # Apply upsample per group.
    # We can achieve this by iterating through groups, or by using group_by().apply()
    # For upsampling, it's often cleaner to do it within an apply if other operations follow
    # or ensure the DataFrame is correctly sorted and then upsample.
    # The `DataFrame.upsample` method implicitly handles groups if the `by` argument is used.
    # However, to correctly create the 'is_missing' mask *before* filling for each group,
    # it's better to handle the upsampling and initial 'is_missing' creation per group,
    # then combine and do the filling.

    # For Polars versions where GroupBy.upsample was removed,
    # you now call upsample on the DataFrame and specify group keys with 'by'.
    # However, this doesn't directly allow creating the 'is_missing' column
    # exactly when needed for each group *before* global filling.
    # A common pattern is to use `group_by().apply()` for complex per-group logic.

    # Let's try a more robust group-wise apply approach:
    def upsample_and_mark_missing_per_group(group_df: pl.DataFrame) -> pl.DataFrame:
        group_df_reindexed = group_df.upsample(
            time_column="date",
            every="1d",
            # We don't need 'by' here as it's already a single group from apply
        )
        # Create 'is_missing' mask *for this group*
        group_df_reindexed = group_df_reindexed.with_columns(
            is_missing=pl.col(
                "close"
            ).is_null()  # 'close' will be null for newly created rows
        )
        return group_df_reindexed

    # Apply this function to each group
    # Note: `group_by().apply()` can be slower than vectorized operations for very large numbers of small groups.
    # If performance becomes an issue, we might need to optimize this further.
    # For now, clarity and correctness are key.
    df_reindexed_list = []
    for _, group_data in df.group_by("ticker_id", maintain_order=True):
        df_reindexed_list.append(upsample_and_mark_missing_per_group(group_data))

    if not df_reindexed_list:  # Handle empty input DataFrame
        return df.clear().with_columns(
            pl.lit(None).cast(pl.Boolean).alias("is_missing")
        )

    df_reindexed = pl.concat(df_reindexed_list)

    # Forward-fill the data for relevant columns with a limit
    # This fill needs to happen per group to avoid bleeding data across tickers.
    fill_cols = [
        "close",
        "open",
        "volume",
        "ticker",
        "industry",
        "sector_id",
        "market_cap",
    ]  # Added market_cap

    # We need to ensure ticker_id (and other static-like columns within a group) are filled first
    # before filling the time-varying ones.
    # 'ticker_id' should already be present from the grouping key, but let's be safe.
    # The `upsample` within `apply` should preserve the `ticker_id` for the group.
    # If not, we might need to explicitly add `pl.col("ticker_id").first().alias("ticker_id")` in the apply.

    # Perform forward fill per group
    df_filled = df_reindexed.with_columns(
        pl.col(fill_cols).forward_fill(limit=max_ffill_days).over("ticker_id")
    )

    # Drop rows where ticker_id is still null (these are gaps at the very start of a series,
    # or if something went wrong with ticker_id propagation)
    df_filled = df_filled.drop_nulls(subset=["ticker_id"])

    # Ensure the 'is_missing' column, which was created *before* ffill, is preserved correctly.
    # And ensure other non-filled columns from the original df are present.
    # The `upsample` should carry over all original columns.

    return df_filled


def create_features_and_targets(df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates all model features and targets.
    - Targets: Log-returns for 1, 5, 20 day horizons.
    - Known-future features: Calendar features.
    - Observed-past features: Returns, Volatility, RSI, MACD, etc.
    """
    print("Creating features and targets...")

    df = df.sort("ticker_id", "date")

    safe_close = pl.when(pl.col("close") <= 1e-6).then(1e-6).otherwise(pl.col("close"))
    log_close = safe_close.log()

    # --- Target Definition ---
    df = df.with_columns(
        (log_close.shift(-1) - log_close).over("ticker_id").alias("target_1d"),
        (log_close.shift(-5) - log_close).over("ticker_id").alias("target_5d"),
        (log_close.shift(-20) - log_close).over("ticker_id").alias("target_20d"),
    )

    # --- ADDED PER YOUR RECOMMENDATION ---
    # Clip obviously impossible returns to prevent outlier explosion.
    # A 25% daily return is already extreme. We nullify anything beyond that.
    MAX_ABS_DAILY_RETURN = 0.25
    print(
        f"Clipping targets with absolute daily log-return > {MAX_ABS_DAILY_RETURN}..."
    )
    df = df.with_columns(
        [
            pl.when(pl.col("target_1d").abs() > MAX_ABS_DAILY_RETURN)
            .then(None)
            .otherwise(pl.col("target_1d"))
            .alias("target_1d"),
            pl.when(pl.col("target_5d").abs() > MAX_ABS_DAILY_RETURN * 5)
            .then(None)
            .otherwise(pl.col("target_5d"))
            .alias("target_5d"),
            pl.when(pl.col("target_20d").abs() > MAX_ABS_DAILY_RETURN * 20)
            .then(None)
            .otherwise(pl.col("target_20d"))
            .alias("target_20d"),
        ]
    )
    # --- END OF ADDED BLOCK ---

    # --- Feature Engineering ---
    # 1. Known-future (calendar) features
    df = df.with_columns(
        day_of_week=pl.col("date").dt.weekday().cast(pl.Float32),
        day_of_month=pl.col("date").dt.day().cast(pl.Float32),
        month=pl.col("date").dt.month().cast(pl.Float32),
        is_quarter_end=(pl.col("date").dt.month().is_in([3, 6, 9, 12]))
        & (pl.col("date").dt.month_end() == pl.col("date")),
    ).with_columns(pl.col("is_quarter_end").cast(pl.Float32))

    if "is_missing" in df.columns:
        df = df.with_columns(pl.col("is_missing").cast(pl.Float32))
    else:
        print("Warning: 'is_missing' column not found before feature engineering.")
        df = df.with_columns(pl.lit(0.0).cast(pl.Float32).alias("is_missing"))

    # 2. Observed-past features
    log_return_1d_base_expr = log_close - log_close.shift(1)

    rolling_volume_mean = pl.col("volume").rolling_mean(window_size=20, min_samples=10)
    rolling_volume_std = pl.col("volume").rolling_std(window_size=20, min_samples=10)

    volume_zscore_expr = (
        pl.when(rolling_volume_std > 1e-6)
        .then((pl.col("volume") - rolling_volume_mean) / rolling_volume_std)
        .otherwise(0.0)
    )

    # --- CORRECTED: Moved `.over()` outside the `pl.when()` clause ---
    is_missing_mask = pl.col("is_missing").cast(pl.Boolean)

    df = df.with_columns(
        log_return_1d=(pl.when(~is_missing_mask).then(log_return_1d_base_expr)).over(
            "ticker_id"
        ),
        log_return_5d=(
            pl.when(~is_missing_mask).then(log_close - log_close.shift(5))
        ).over("ticker_id"),
        log_return_20d=(
            pl.when(~is_missing_mask).then(log_close - log_close.shift(20))
        ).over("ticker_id"),
        volume_zscore_20d=(pl.when(~is_missing_mask).then(volume_zscore_expr)).over(
            "ticker_id"
        ),
        volatility_20d=(
            pl.when(~is_missing_mask).then(
                log_return_1d_base_expr.rolling_std(window_size=20, min_samples=10)
            )
        ).over("ticker_id"),
        skew_20d=(
            pl.when(~is_missing_mask).then(
                log_return_1d_base_expr.rolling_skew(window_size=20)
            )
        ).over("ticker_id"),
        kurtosis_20d=(
            pl.when(~is_missing_mask).then(
                log_return_1d_base_expr.rolling_kurtosis(window_size=20)
            )
        ).over("ticker_id"),
        price_change=(pl.when(~is_missing_mask).then(safe_close.diff(1))).over(
            "ticker_id"
        ),
    )

    # RSI calculation
    df = df.with_columns(
        gain=pl.when(pl.col("price_change") > 0)
        .then(pl.col("price_change"))
        .otherwise(0.0)
        .alias("gain"),
        loss=pl.when(pl.col("price_change") < 0)
        .then(-pl.col("price_change"))
        .otherwise(0.0)
        .alias("loss"),
    )

    avg_gain_expr = pl.col("gain").ewm_mean(alpha=1 / 14, min_samples=10)
    avg_loss_expr = pl.col("loss").ewm_mean(alpha=1 / 14, min_samples=10)

    df = df.with_columns(
        avg_gain=avg_gain_expr.over("ticker_id"),
        avg_loss=avg_loss_expr.over("ticker_id"),
    )

    rs_expr = (
        pl.when(pl.col("avg_loss") > 1e-6)
        .then(pl.col("avg_gain") / pl.col("avg_loss"))
        .otherwise(pl.when(pl.col("avg_gain") > 1e-6).then(100.0).otherwise(1.0))
    )

    df = df.with_columns(
        rsi_14d=(
            pl.when(~is_missing_mask).then(100.0 - (100.0 / (1.0 + rs_expr)))
        ).alias("rsi_14d")
    )

    # MACD
    ema_12_expr = safe_close.ewm_mean(
        span=12, min_samples=12 - 1 if (12 - 1) > 0 else 1
    )
    ema_26_expr = safe_close.ewm_mean(
        span=26, min_samples=26 - 1 if (26 - 1) > 0 else 1
    )

    # CORRECTED: Apply the when/then logic within the `.over()` context
    df = df.with_columns(
        macd_base=(pl.when(~is_missing_mask).then(ema_12_expr - ema_26_expr)).over(
            "ticker_id"
        )
    )

    macd_signal_expr = pl.col("macd_base").ewm_mean(
        span=9, min_samples=9 - 1 if (9 - 1) > 0 else 1
    )
    df = df.with_columns(
        macd_signal_base=(pl.when(~is_missing_mask).then(macd_signal_expr)).over(
            "ticker_id"
        )
    )

    df = df.rename({"macd_base": "macd", "macd_signal_base": "macd_signal"})
    # --- END CORRECTION ---

    # --- Final Cleanup ---

    # 1. Drop intermediate columns
    intermediate_cols = ["price_change", "gain", "loss", "avg_gain", "avg_loss"]
    cols_to_drop_now = [col for col in intermediate_cols if col in df.columns]
    if cols_to_drop_now:
        df = df.drop(cols_to_drop_now)

    # 2. Convert any generated infinities or NaNs in float columns to proper nulls
    float_cols = [
        col_name
        for col_name, dtype in df.schema.items()
        if dtype in [pl.Float32, pl.Float64]
    ]
    for col_name in float_cols:
        df = df.with_columns(
            pl.when(pl.col(col_name).is_nan() | pl.col(col_name).is_infinite())
            .then(None)
            .otherwise(pl.col(col_name))
            .alias(col_name)
        )

    # --- START: NEW, MORE ROBUST NULL HANDLING ---

    # 3. Drop rows only where essential targets are null.
    # This keeps the maximum amount of historical data for feature generation.
    target_cols = ["target_1d", "target_5d", "target_20d"]
    df = df.drop_nulls(subset=target_cols)
    print(f"Shape after dropping rows with null targets: {df.shape}")

    # 4. Forward-fill features to handle nulls at the start of each series.
    # Then fill any remaining nulls (at the very beginning) with 0.
    # Exclude IDs, dates, and targets from this fill.
    feature_cols = [
        col
        for col in df.columns
        if col not in ["ticker", "date", "ticker_id", "sector_id"] + target_cols
    ]
    df = df.with_columns(
        pl.col(feature_cols).forward_fill().over("ticker_id")
    ).with_columns(pl.col(feature_cols).fill_null(0.0))
    print(f"Shape after forward-filling and zero-filling features: {df.shape}")

    # 5. As a final safety check, drop any row that might still have a null value.
    # This should now drop very few, if any, rows.
    df = df.drop_nulls()

    # --- END: NEW, MORE ROBUST NULL HANDLING ---

    # 6. Sort the data
    df = df.sort("ticker_id", "date")

    print(
        f"Feature creation complete. After NaN/inf handling AND FINAL drop_nulls(), shape: {df.shape}"
    )
    if df.height == 0:
        raise ValueError(
            "All data was dropped after feature engineering and NaN/inf handling. Check data quality and feature logic."
        )

    # 7. Create the final time_idx
    df = df.with_columns(
        time_idx=(pl.col("date").rank("ordinal").over("ticker_id") - 1)
    )

    return df


if __name__ == "__main__":
    DATA_DIR = Path("data")
    RAW_DATA_PATH = DATA_DIR / "raw" / "stock_data.csv"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    PROCESSED_PARQUET_PATH = PROCESSED_DATA_DIR / "processed_data.parquet"

    print("--- Starting Data Pipeline ---")
    df_initial = load_and_clean_data(RAW_DATA_PATH)
    initial_rows = df_initial.height

    df_mapped = create_mappings(df_initial, PROCESSED_DATA_DIR)
    df_reindexed = reindex_and_fill_gaps(df_mapped)
    df_final = create_features_and_targets(df_reindexed)

    print("\n--- Final Processed DataFrame ---")
    if df_final.height > 0:
        print(df_final.head())

        # --- CORRECTED DEBUGGING STATEMENTS ---
        final_rows = df_final.height
        retention = (final_rows / initial_rows) * 100 if initial_rows > 0 else 0
        print(f"\nData retention: {final_rows}/{initial_rows} ({retention:.1f}%)")

        # Check for any remaining nulls (should be zero) using idiomatic Polars
        total_nulls = df_final.null_count().select(pl.sum_horizontal("*")).item()
        print(f"Total null values in final data: {total_nulls}")
        if total_nulls > 0:
            print("Warning: Null values detected in final DataFrame!")
            print(df_final.null_count())
        # --- END CORRECTION ---

        print(f"\nSaving final processed data to {PROCESSED_PARQUET_PATH}...")
        df_final.write_parquet(PROCESSED_PARQUET_PATH)
    else:
        print("No data left after processing. Parquet file not saved.")
    print("--- Data Pipeline Complete ---")


===== src/market_prediction_workbench/model.py =====
# src/market_prediction_workbench/model.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from pytorch_forecasting.models.temporal_fusion_transformer._tft import (
    TemporalFusionTransformer,
)
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss
import numpy as np

# Added for the new __init__ logic
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd


class TemporalFusionTransformerWithDevice(TemporalFusionTransformer):
    def get_attention_mask(self, *args, **kwargs) -> torch.Tensor:
        if len(args) == 3 and all(isinstance(x, (int, torch.Tensor)) for x in args):
            raw_bs, raw_enc, raw_dec = args

            def to_int(x):
                return int(x.item()) if torch.is_tensor(x) else int(x)

            batch_size, L_enc, L_dec = to_int(raw_bs), to_int(raw_enc), to_int(raw_dec)
            mask_device = self.device
        else:
            raw_enc = kwargs.get("encoder_lengths", kwargs.get("max_encoder_length"))
            raw_dec = kwargs.get("decoder_lengths", kwargs.get("max_prediction_length"))
            if raw_enc is None or raw_dec is None:
                raise RuntimeError(
                    f"Could not parse lengths from args={args}, kwargs={kwargs}"
                )
            if torch.is_tensor(raw_enc) and raw_enc.dim() >= 1:
                batch_size = raw_enc.shape[0]
            elif torch.is_tensor(raw_dec) and raw_dec.dim() >= 1:
                batch_size = raw_dec.shape[0]
            else:
                raise RuntimeError(
                    f"Cannot infer batch_size (got {raw_enc}, {raw_dec})"
                )

            def to_int(x):
                return (
                    int(x[0].item())
                    if torch.is_tensor(x) and x.numel() > 1
                    else (int(x.item()) if torch.is_tensor(x) else int(x))
                )

            L_enc, L_dec = to_int(raw_enc), to_int(raw_dec)
            if torch.is_tensor(raw_enc):
                mask_device = raw_enc.device
            elif torch.is_tensor(raw_dec):
                mask_device = raw_dec.device
            else:
                mask_device = self.device

        decoder_mask = torch.triu(
            torch.ones((L_dec, L_dec), dtype=torch.bool, device=mask_device), diagonal=1
        )
        decoder_mask = decoder_mask.unsqueeze(0).expand(batch_size, L_dec, L_dec)
        encoder_mask = torch.zeros((L_dec, L_enc), dtype=torch.bool, device=mask_device)
        encoder_mask = encoder_mask.unsqueeze(0).expand(batch_size, L_dec, L_enc)
        return torch.cat([decoder_mask, encoder_mask], dim=-1)


class GlobalTFT(pl.LightningModule):
    def __init__(
        self,
        model_specific_params: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        timeseries_dataset: TimeSeriesDataSet | None = None,
        timeseries_dataset_params: dict | None = None,
    ):
        super().__init__()

        if timeseries_dataset is None and timeseries_dataset_params is None:
            raise ValueError(
                "Either `timeseries_dataset` or `timeseries_dataset_params` must be provided."
            )

        # If loading from checkpoint, timeseries_dataset might be None.
        # Reconstruct a skeleton dataset from params to initialize the model.
        if timeseries_dataset is None:
            # This allows loading a model without having to load the data first.
            # The actual data is needed for the dataloaders, but not for model architecture.
            timeseries_dataset = TimeSeriesDataSet.from_parameters(
                timeseries_dataset_params, pd.DataFrame(), predict=True
            )

        # The original __init__ logic can proceed from here.
        num_targets = len(timeseries_dataset.target_names)

        loss_instance = model_specific_params.get("loss")
        quantiles = (
            loss_instance.quantiles if hasattr(loss_instance, "quantiles") else [0.5]
        )
        output_size = len(quantiles)
        if "output_size" not in model_specific_params:
            model_specific_params["output_size"] = output_size

        # We save the parameters, not the full dataset object.
        # The model_specific_params might contain non-serializable objects (like the loss function instance)
        # so we clean it for saving.

        if isinstance(model_specific_params["output_size"], int):
            model_specific_params["output_size"] = [
                model_specific_params["output_size"]
            ] * num_targets

        init_model_params_copy = model_specific_params.copy()

        cleaned_model_params = {
            k: v
            for k, v in init_model_params_copy.items()
            if not isinstance(v, (nn.Module, torch.nn.modules.loss._Loss))
        }

        # Save hyperparameters for checkpointing. This is crucial for reloading.
        self.save_hyperparameters(
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "model_specific_params": cleaned_model_params,
                "timeseries_dataset_params": timeseries_dataset.get_parameters(),
            }
        )

        self.model = TemporalFusionTransformerWithDevice.from_dataset(
            timeseries_dataset,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **init_model_params_copy,  # Pass original params with loss object
        )

        dataset_max_pred_len = timeseries_dataset.max_prediction_length
        current_loss_module = self.model.loss
        if isinstance(current_loss_module, MultiLoss):
            for metric in current_loss_module.metrics:
                if (
                    hasattr(metric, "max_prediction_length")
                    and metric.max_prediction_length is None
                ):
                    metric.max_prediction_length = dataset_max_pred_len
                    print(
                        f"DEBUG GlobalTFT.__init__: Set metric {type(metric).__name__}.max_prediction_length to {dataset_max_pred_len}"
                    )
        elif hasattr(current_loss_module, "max_prediction_length"):
            if current_loss_module.max_prediction_length is None:
                current_loss_module.max_prediction_length = dataset_max_pred_len
                print(
                    f"DEBUG GlobalTFT.__init__: Set self.model.loss {type(current_loss_module).__name__}.max_prediction_length to {dataset_max_pred_len}"
                )
            # Debug print for confirmation
            print(
                f"DEBUG GlobalTFT.__init__: self.model.loss ({type(current_loss_module).__name__}).max_prediction_length = {current_loss_module.max_prediction_length}"
            )

        else:
            print(
                f"DEBUG GlobalTFT.__init__: self.model.loss ({type(current_loss_module).__name__}) does not have max_prediction_length attribute."
            )

        # Temporary override to simplify debugging:
        self.model.logging_metrics = nn.ModuleList([])
        print("DEBUG: model.logging_metrics has been cleared for debugging.")

    # ... _process_input_data, _prepare_target_tensor, _prepare_scale_tensor ...
    def _process_input_data(self, data_element):
        # print(f"DEBUG _process_input_data: received type {type(data_element)}")
        if torch.is_tensor(data_element):
            # print(f"DEBUG _process_input_data: is_tensor, shape {data_element.shape}")
            return data_element.to(device=self.device, dtype=torch.float32)
        elif isinstance(data_element, np.ndarray):
            # print(f"DEBUG _process_input_data: is_ndarray, dtype {data_element.dtype}, shape {data_element.shape}")
            if data_element.dtype == object:
                if (
                    data_element.ndim == 0
                    and hasattr(data_element.item(), "to")
                    and hasattr(data_element.item(), "device")
                ):
                    return data_element.item().to(
                        device=self.device, dtype=torch.float32
                    )
                try:
                    processed_list = [
                        self._process_input_data(el) for el in data_element
                    ]
                    return (
                        torch.stack(processed_list)
                        if processed_list
                        else torch.empty(0, dtype=torch.float32, device=self.device)
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to process numpy array of dtype 'object'. Shape: {data_element.shape}. "
                        f"First element type: {type(data_element.flat[0]) if data_element.size > 0 else 'N/A'}. Error: {e}"
                    ) from e
            else:
                return torch.from_numpy(data_element).to(
                    device=self.device, dtype=torch.float32
                )
        elif isinstance(data_element, (list, tuple)):
            # print(f"DEBUG _process_input_data: is_list_or_tuple, len {len(data_element)}")
            if not data_element:
                return torch.empty(0, dtype=torch.float32, device=self.device)
            processed_elements = [
                self._process_input_data(item) for item in data_element
            ]
            try:
                return torch.stack(processed_elements)
            except RuntimeError as e:
                try:
                    return torch.tensor(
                        data_element, dtype=torch.float32, device=self.device
                    )
                except Exception as e_tensor:
                    shapes = [
                        (
                            f"{type(pe)}:{pe.shape}"
                            if torch.is_tensor(pe)
                            else str(type(pe))
                        )
                        for pe in processed_elements
                    ]
                    raise RuntimeError(
                        f"Failed to stack OR create tensor from processed elements from list/tuple. Original type: {type(data_element)}. "
                        f"Processed element types/shapes: {shapes}. Stack error: {e}. Tensor creation error: {e_tensor}"
                    ) from e_tensor
        else:
            try:
                # print(f"DEBUG _process_input_data: is_other_scalar, value {data_element}")
                return torch.tensor(
                    data_element, dtype=torch.float32, device=self.device
                )
            except Exception as e:
                raise TypeError(
                    f"Unsupported type {type(data_element)} for _process_input_data. Error: {e}"
                ) from e

    def _prepare_target_tensor(self, raw_target_data):
        target = self._process_input_data(raw_target_data)

        # This was part of the original file and needs to be adapted for lists
        if isinstance(target, list):
            return [t.unsqueeze(1) if t.ndim == 1 else t for t in target]

        if target.ndim == 2:
            target = target.unsqueeze(1)  # Add time dimension
        return target

    def _prepare_scale_tensor(self, raw_scale_data, target_shape):
        scale = self._process_input_data(raw_scale_data)
        if (
            scale.ndim == 1 and scale.shape[0] > 0 and scale.shape[0] == target_shape[0]
        ):  # (B,)
            scale = scale.unsqueeze(-1)  # (B,1)
        return scale

    # ... forward, training_step, validation_step, configure_optimizers (no changes here) ...
    def forward(self, x_batch):
        return self.model(x_batch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        target, _ = y  # y is a tuple of (target, scale)

        # --- THE CORRECT FIX for both AttributeError and IndexError ---
        # The loss function expects a time dimension for each target.
        # Since max_prediction_horizon=1, targets from the dataloader are 1D.
        # We must add a time dimension of size 1.
        if isinstance(target, list):
            # If multi-target, reshape each tensor in the list.
            # Shape changes from [(B,), (B,), ...] to [(B, 1), (B, 1), ...]
            reshaped_target = [t.unsqueeze(1) for t in target]
        else:
            # If single-target, reshape the single tensor.
            # Shape changes from (B,) to (B, 1)
            reshaped_target = target.unsqueeze(1)
        # --- END FIX ---

        loss_val = self.model.loss(out.prediction, reshaped_target)
        bs = (
            reshaped_target[0].size(0)
            if isinstance(reshaped_target, list)
            else reshaped_target.size(0)
        )
        self.log(
            "train_loss",
            loss_val,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )
        return loss_val

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        target, _ = y

        # --- THE CORRECT FIX (Applied to validation as well) ---
        if isinstance(target, list):
            reshaped_target = [t.unsqueeze(1) for t in target]
        else:
            reshaped_target = target.unsqueeze(1)
        # --- END FIX ---

        loss_val = self.model.loss(out.prediction, reshaped_target)
        self.log("val_loss", loss_val)
        return loss_val

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            fused=True if torch.cuda.is_available() else False,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            factor=0.5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }





---
INPUT 3: SECOND HALF OF THE EXISTING CODE
---


===== src/market_prediction_workbench/evaluate.py =====
# src/market_prediction_workbench/evaluate.py
import json
import sys
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from market_prediction_workbench.model import GlobalTFT

# -----------------------------------------------------------------------------#
# Helpers                                                                       #
# -----------------------------------------------------------------------------#


def _cfg_list(val) -> List[str]:
    """Return a list of strings from a Hydra list / scalar / None."""
    if val is None:
        return []
    if isinstance(val, (str, int, float)):
        return [str(val)]
    if isinstance(val, (list, ListConfig)):
        return [str(v) for v in val]
    raise TypeError(f"Unsupported cfg node type: {type(val)}")


def _safe_parse_val_loss(stem: str) -> float:
    """
    Extract `val_loss=<float>` from a checkpoint stem.
    Returns +inf when the pattern is missing or malformed so that `.sort()` can
    still work without crashing.
    """
    if "val_loss=" not in stem:
        return float("inf")
    try:
        return float(stem.split("val_loss=")[-1].split("-")[0])
    except Exception:
        return float("inf")


def _move_to_device(obj, device):
    """Recursively send tensors to the selected device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


# evaluate.py
# -----------------------------------------------------------------------
# REPLACE the whole inverse_transform_with_groups(...) helper with this
# -----------------------------------------------------------------------

# evaluate.py  – replace _inverse_with_groups with this version
# evaluate.py  ------------------------------------------------------------


def _inverse_with_groups(
    data: torch.Tensor, normalizer, groups: torch.Tensor
) -> torch.Tensor:
    """
    Robust inverse transform that works on every PF version:
       • GroupNormalizer        (any transformation / any release)
       • MultiNormalizer        (recursion)
       • Other normalisers      (delegate)
    """
    # ------------------------------------------------------------------ #
    # 1) GroupNormalizer                                                #
    # ------------------------------------------------------------------ #
    if isinstance(normalizer, GroupNormalizer):
        # figure out which keyword, if any, the current build understands
        sig = inspect.signature(normalizer.inverse_transform)
        if "group_ids" in sig.parameters:
            kw = "group_ids"
        elif "groups" in sig.parameters:
            kw = "groups"
        elif "target_scale" in sig.parameters:
            kw = "target_scale"
        elif "scale" in sig.parameters:
            kw = "scale"
        else:
            kw = None  # positional only

        # obtain µ (location) and σ (scale) for every sample in the batch
        g = groups[:, 0].cpu().numpy()  # [B]
        scale = torch.as_tensor(
            normalizer.get_parameters(g),  # [B, 2] (loc, scale)
            dtype=data.dtype,
            device=data.device,
        )

        # try the built-in inverse first …
        try:
            if kw is None:
                return normalizer.inverse_transform(data, scale)
            else:
                return normalizer.inverse_transform(data, **{kw: scale})

        # … fall back to manual µ+σ·ŷ when NotImplementedError is raised
        except NotImplementedError:
            loc = scale[:, 0]
            sigm = scale[:, 1]
            while loc.dim() < data.dim():
                loc = loc.unsqueeze(1)
            while sigm.dim() < data.dim():
                sigm = sigm.unsqueeze(1)
            return data * sigm + loc

    # ------------------------------------------------------------------ #
    # 2) MultiNormalizer                                                 #
    # ------------------------------------------------------------------ #
    if isinstance(normalizer, MultiNormalizer):
        parts = [
            _inverse_with_groups(data[..., i], sub, groups)
            for i, sub in enumerate(normalizer.normalizers)
        ]
        return torch.stack(parts, dim=-1)

    # ------------------------------------------------------------------ #
    # 3) Anything else                                                   #
    # ------------------------------------------------------------------ #
    out = normalizer.inverse_transform(data)
    return torch.as_tensor(out, dtype=data.dtype, device=data.device)


def inverse_transform_with_groups(
    data: torch.Tensor, normalizer, groups: torch.Tensor
) -> torch.Tensor:
    """
    Inverse-transform a tensor normalized by:
      - GroupNormalizer  => manually invert using get_parameters
      - MultiNormalizer  => recurse into each sub-normalizer
      - others           => call built-in inverse_transform
    """
    # Handle GroupNormalizer
    if isinstance(normalizer, GroupNormalizer):
        group_ids = groups[:, 0].cpu().numpy()
        params = normalizer.get_parameters(group_ids)
        mus = torch.from_numpy(params[:, 0]).unsqueeze(1).to(data.device)
        return (data + 1) * mus

    # Handle MultiNormalizer
    if isinstance(normalizer, MultiNormalizer):
        # Special case for single-element data (single target)
        if data.dim() == 1 or (data.dim() == 2 and data.shape[1] == 1):
            # Process first target only
            return inverse_transform_with_groups(
                data, normalizer.normalizers[0], groups
            )

        # Multi-target case - process each target separately
        parts = [
            inverse_transform_with_groups(data[..., i], subnorm, groups)
            for i, subnorm in enumerate(normalizer.normalizers)
        ]
        return torch.stack(parts, dim=-1)

    # Handle other normalizers
    arr = normalizer.inverse_transform(data.cpu().numpy())
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    return arr.to(data.device)


# -----------------------------------------------------------------------------#
# Inference                                                                    #
# -----------------------------------------------------------------------------#


def run_inference(
    model: GlobalTFT, loader: DataLoader, cfg: DictConfig, dataset: TimeSeriesDataSet
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Run model on the dataloader and collect predictions with inverse transform."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    group_id_col = _cfg_list(cfg.data.group_ids)[0]
    decoder = dataset.categorical_encoders[group_id_col]

    preds_norm_list, trues_norm_list = [], []
    tickers_all, t_idx_all = [], []
    groups_all = []  # Store groups for inverse transform

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Running inference"):
            x = _move_to_device(x, device)

            # normalized target
            if isinstance(y, (list, tuple)):
                target_norm = y[0]
            else:
                target_norm = y

            # forward
            output_norm = model(x).prediction
            if isinstance(output_norm, list):
                output_norm = torch.stack(output_norm, dim=2)
            else:
                output_norm = output_norm.unsqueeze(2)

            if isinstance(target_norm, list):
                target_norm = torch.stack(target_norm, dim=2)
            else:
                target_norm = target_norm.unsqueeze(2)

            # collect on CPU
            preds_norm_list.append(output_norm.cpu())
            trues_norm_list.append(target_norm.cpu())

            # Store groups for later use in inverse transform
            groups_all.append(x["groups"].cpu())

            # decode tickers & times
            encoded = x["groups"][:, 0].cpu().numpy()
            tickers_all.extend(int(s) for s in decoder.inverse_transform(encoded))
            t_idx_all.append(x["decoder_time_idx"].cpu().numpy())

    # concatenate
    preds_norm = torch.cat(preds_norm_list, dim=0)  # [B, H, T, Q]
    trues_norm = torch.cat(trues_norm_list, dim=0)  # [B, H, T]
    groups = torch.cat(groups_all, dim=0)  # [B, G] where G is number of group columns
    tickers = np.array(tickers_all)
    time_idx = np.concatenate(t_idx_all, axis=0)  # [B, H]

    # Handle different types of normalizers
    normalizer = dataset.target_normalizer

    # Inverse-transform predictions
    preds_dec = []
    num_targets = preds_norm.shape[2]

    for i in range(num_targets):
        target_preds = []
        for q in range(preds_norm.shape[3]):
            pred_data = preds_norm[:, :, i, q]

            # Get the appropriate normalizer
            if isinstance(normalizer, MultiNormalizer) and i < len(
                normalizer.normalizers
            ):
                norm = normalizer.normalizers[i]
            else:
                norm = normalizer

            # Handle GroupNormalizer specifically
            if isinstance(norm, GroupNormalizer):
                decoded = _inverse_with_groups(pred_data, norm, groups)
            else:
                decoded = norm.inverse_transform(pred_data)
                if not isinstance(decoded, torch.Tensor):
                    decoded = torch.tensor(decoded)

            target_preds.append(decoded)

        preds_dec.append(torch.stack(target_preds, dim=-1))

    # Stack targets: [B, H, T, Q]
    preds_dec = torch.stack(preds_dec, dim=2)

    # Inverse-transform true values
    trues_dec = _inverse_with_groups(trues_norm, normalizer, groups)

    # Extract first horizon predictions
    preds_h1 = preds_dec[:, 0]  # [B, T, Q]
    trues_h1 = trues_dec[:, 0]  # [B, T]

    # Build result dictionaries
    pred_dict, true_dict = {}, {"ticker": tickers, "time_idx": time_idx[:, 0]}
    short_names = [t.replace("target_", "") for t in _cfg_list(cfg.data.target)]

    for i, name in enumerate(short_names):
        pred_dict[f"{name}_lower"] = preds_h1[:, i, 0].numpy()
        pred_dict[f"{name}"] = preds_h1[:, i, 1].numpy()
        pred_dict[f"{name}_upper"] = preds_h1[:, i, 2].numpy()
        true_dict[name] = trues_h1[:, i].numpy()

    return pred_dict, true_dict, short_names


# -----------------------------------------------------------------------------#
# Evaluation Metrics                                                           #
# -----------------------------------------------------------------------------#


def evaluate(preds: Dict, trues: Dict, short_names: List[str]) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    metrics = {}

    for name in short_names:
        pred = preds[f"{name}"]  # median prediction
        true = trues[name]

        # Remove any NaN values
        mask = ~(np.isnan(pred) | np.isnan(true))
        pred = pred[mask]
        true = true[mask]

        if len(pred) == 0:
            continue

        # Calculate metrics
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))

        # Coverage: what fraction of true values fall within prediction intervals
        lower = preds[f"{name}_lower"][mask]
        upper = preds[f"{name}_upper"][mask]
        coverage = np.mean((true >= lower) & (true <= upper))

        metrics[f"{name}_mae"] = mae
        metrics[f"{name}_rmse"] = rmse
        metrics[f"{name}_coverage_90"] = coverage

    return metrics


# -----------------------------------------------------------------------------#
# Visualisation & Output                                                       #
# -----------------------------------------------------------------------------#


def _safe_ticker_id(df: pl.DataFrame, ticker: str) -> int | None:
    try:
        return df.filter(pl.col("ticker") == ticker)["ticker_id"].item()
    except Exception:
        return None


def plot_preds(preds, trues, out_dir, ticker_map, sample_tickers, short_tgt_names):
    out_dir.mkdir(parents=True, exist_ok=True)
    ticker_csv_dir = out_dir / "ticker_predictions_for_plot"
    ticker_csv_dir.mkdir(parents=True, exist_ok=True)

    df_plot = pd.DataFrame(trues)
    for key, val in preds.items():
        df_plot[f"p_{key}"] = val

    for tk in sample_tickers:
        tid = _safe_ticker_id(ticker_map, tk)
        if tid is None:
            print(f"'{tk}' not found in ticker_map – skipping.")
            continue

        df_ticker = df_plot[df_plot["ticker"] == tid].sort_values("time_idx")

        ticker_csv_path = ticker_csv_dir / f"{tk}_predictions.csv"
        df_ticker.to_csv(ticker_csv_path, index=False)
        print(f"Saved prediction data for '{tk}' to {ticker_csv_path}")

        if len(df_ticker) < 2:
            print(f"Ticker '{tk}' has <2 predictions – skipping plot.")
            continue

        plt.figure(figsize=(15, 4 * len(short_tgt_names)))
        for i, name in enumerate(short_tgt_names):
            plt.subplot(len(short_tgt_names), 1, i + 1)
            plt.plot(df_ticker["time_idx"], df_ticker[name], "b-", label="Actual")
            plt.plot(
                df_ticker["time_idx"],
                df_ticker[f"p_{name}"],
                "r--",
                label="Pred (median)",
            )
            plt.fill_between(
                df_ticker["time_idx"],
                df_ticker[f"p_{name}_lower"],
                df_ticker[f"p_{name}_upper"],
                alpha=0.2,
                color="orange",
                label="90% PI",
            )
            plt.title(f"{tk} – {name.upper()} horizon")
            plt.ylabel("Log-return")
            plt.legend()

        plt.xlabel("Time index")
        plt.tight_layout()
        plt.savefig(out_dir / f"{tk}_timeseries.png")
        plt.close()
        print(f"Successfully generated plot for {tk}.")


def save_output(metrics: Dict, preds: Dict, trues: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w") as fp:
        json.dump({k: float(v) for k, v in metrics.items()}, fp, indent=2)

    pd.DataFrame({**trues, **preds}).to_csv(out_dir / "predictions.csv", index=False)
    print(f"\nResults written to {out_dir}")


# -----------------------------------------------------------------------------#
# Entry-point                                                                  #
# -----------------------------------------------------------------------------#


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    log_dir = Path(cfg.paths.log_dir)

    ckpts = list(log_dir.glob("**/*.ckpt"))
    if not ckpts:
        sys.exit(f"No .ckpt files found under {log_dir}")

    run_dirs = {
        p.parent.parent if p.parent.name == "checkpoints" else p.parent for p in ckpts
    }
    latest_run = max(
        run_dirs, key=lambda d: max(x.stat().st_mtime for x in d.glob("**/*.ckpt"))
    )
    print(f"Identified latest run directory: {latest_run}")

    cand_ckpts = list((latest_run / "checkpoints").glob("*.ckpt")) or list(
        latest_run.glob("*.ckpt")
    )
    cand_ckpts.sort(key=lambda p: _safe_parse_val_loss(p.stem))
    best = next((p for p in cand_ckpts if "best" in p.stem), None) or cand_ckpts[0]
    print(f"\nEvaluating checkpoint: {best}")

    parquet_path = Path(cfg.paths.processed_data_file)
    if not parquet_path.exists():
        sys.exit(f"Processed data file not found at {parquet_path}")

    print("Loading full processed dataset to match model architecture...")
    df_full = pd.read_parquet(parquet_path)

    cp = torch.load(best, map_location="cpu", weights_only=False)
    ds_params = cp["hyper_parameters"]["timeseries_dataset_params"]

    time_idx_col = ds_params["time_idx"]
    df_full[time_idx_col] = df_full[time_idx_col].astype(int)

    all_cat_cols = (
        ds_params.get("static_categoricals", [])
        + ds_params.get("time_varying_known_categoricals", [])
        + ds_params.get("time_varying_unknown_categoricals", [])
    )
    for col in all_cat_cols:
        if col in df_full.columns:
            df_full[col] = df_full[col].astype(str)
    print(f"Ensured '{time_idx_col}' is int and categoricals are strings.")

    full_dataset = TimeSeriesDataSet.from_parameters(ds_params, df_full, predict=False)
    print(f"Recreated full dataset with {len(full_dataset)} samples.")

    model = GlobalTFT.load_from_checkpoint(
        best, timeseries_dataset=full_dataset, map_location="cpu"
    )
    print("Model loaded successfully.")

    tickers_to_evaluate = _cfg_list(cfg.evaluate.sample_tickers)
    if not tickers_to_evaluate:
        sys.exit("No tickers specified in cfg.evaluate.sample_tickers. Aborting.")

    cfg_yaml = next(latest_run.glob("**/.hydra/config.yaml"), None)
    run_cfg = OmegaConf.load(cfg_yaml) if cfg_yaml else cfg
    group_id_col = _cfg_list(run_cfg.data.group_ids)[0]

    ticker_map_path = Path(cfg.paths.data_dir) / "processed" / "ticker_map.parquet"
    ticker_map = pl.read_parquet(ticker_map_path) if ticker_map_path.exists() else None

    if ticker_map is None:
        sys.exit("ticker_map.parquet not found. Cannot filter by ticker name.")

    filtered_map = ticker_map.filter(pl.col("ticker").is_in(tickers_to_evaluate))
    ticker_ids_to_evaluate_str = [
        str(i) for i in filtered_map.select("ticker_id").to_series().to_list()
    ]
    if not ticker_ids_to_evaluate_str:
        sys.exit(f"Could not find any of {tickers_to_evaluate} in the ticker_map.")

    eval_dataset = full_dataset.filter(
        lambda x: x[group_id_col].isin(ticker_ids_to_evaluate_str)
    )
    print(
        f"Filtered dataset to {len(eval_dataset)} samples for tickers: {tickers_to_evaluate}"
    )

    loader = eval_dataset.to_dataloader(
        train=False,
        batch_size=cfg.evaluate.batch_size,
        shuffle=False,
        num_workers=cfg.evaluate.num_workers,
    )
    print("Created dataloader for evaluation.")

    preds, trues, short_tgt_names = run_inference(model, loader, run_cfg, full_dataset)
    metrics = evaluate(preds, trues, short_tgt_names)

    print("\n--- Metrics for evaluated tickers ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    out_dir = Path(cfg.paths.log_dir) / "evaluation" / latest_run.name
    plot_preds(
        preds,
        trues,
        out_dir,
        ticker_map,
        tickers_to_evaluate,
        short_tgt_names,
    )
    save_output(metrics, preds, trues, out_dir)


if __name__ == "__main__":
    main()


===== tests/__init__.py =====


===== tests/test_data.py =====
# tests/test_data.py
import pytest
import polars as pl
from pathlib import Path
import torch

# Import from our source code
from market_prediction_workbench.data import (
    load_and_clean_data,
    create_mappings,
    reindex_and_fill_gaps,
    create_features_and_targets,
    MarketDataset,
    DataConfig,
)


# Use a fixture to run the full pipeline once for all tests in this file
@pytest.fixture(scope="module")
def processed_data() -> pl.DataFrame:
    """Runs the full data pipeline and returns the processed DataFrame."""
    # This points to the root of the project where `poetry run pytest` is called
    data_dir = Path("data")
    raw_path = data_dir / "raw" / "stock_data.csv"
    processed_dir = data_dir / "processed"

    # Make sure the raw data exists
    if not raw_path.exists():
        pytest.skip(
            f"Raw data file not found at {raw_path}, skipping integration test."
        )

    df = load_and_clean_data(raw_path)
    df = create_mappings(df, processed_dir)
    df = reindex_and_fill_gaps(df)
    df_final = create_features_and_targets(df)
    return df_final


@pytest.fixture(scope="module")
def data_config() -> DataConfig:
    """Provides a standard DataConfig for tests."""
    return DataConfig(
        static_categoricals=["ticker_id", "sector_id"],
        static_reals=[],
        time_varying_known_reals=[
            "day_of_week",
            "day_of_month",
            "month",
            "is_quarter_end",
        ],
        time_varying_unknown_reals=[
            "log_return_1d",
            "log_return_5d",
            "log_return_20d",
            "volume_zscore_20d",
            "volatility_20d",
            "rsi_14d",
            "macd",
        ],
        target_columns=["target_1d", "target_5d", "target_20d"],
        lookback_days=120,
    )


def test_dataset_creation(processed_data, data_config):
    """Tests if the MarketDataset can be instantiated without errors."""
    dataset = MarketDataset(data=processed_data, config=data_config)
    assert len(dataset) > 0, "Dataset should have at least one valid sample"


def test_dataset_item_shape_and_no_nans(processed_data, data_config):
    """
    Tests that a single item from the dataset has the correct shape, type,
    and contains no NaN values.
    """
    dataset = MarketDataset(data=processed_data, config=data_config)
    sample = dataset[0]  # Get the first sample

    # Check types
    assert isinstance(sample, dict)

    # Check for NaNs
    for key, tensor in sample.items():
        assert isinstance(
            tensor, torch.Tensor
        ), f"Value for key '{key}' is not a tensor"
        if torch.is_floating_point(tensor):
            assert not torch.isnan(tensor).any(), f"NaN found in tensor '{key}'"

    # Check shapes
    lookback = data_config.lookback_days
    assert sample["x_cat"].shape == (len(data_config.static_categoricals),)
    assert sample["x_known_reals"].shape == (
        lookback,
        len(data_config.time_varying_known_reals),
    )
    assert sample["x_unknown_reals"].shape == (
        lookback,
        len(data_config.time_varying_unknown_reals),
    )
    assert sample["y"].shape == (len(data_config.target_columns),)


===== tests/test_model.py =====
# tests/test_model.py
import pytest
import torch
import pandas as pd
import numpy as np
from collections import Counter

from market_prediction_workbench.model import GlobalTFT
from market_prediction_workbench.data import (
    MarketDataset,
    DataConfig,
)  # Assuming MarketDataset is for older custom dataset

# For TFT tests, we need pytorch-forecasting's TimeSeriesDataSet
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from torch.utils.data import WeightedRandomSampler  # for test_balanced_sampler


# Fixture for DataConfig, similar to test_data.py
@pytest.fixture(scope="module")
def data_config_for_tests() -> DataConfig:
    """Provides a standard DataConfig for model tests."""
    return DataConfig(
        static_categoricals=["ticker_id", "sector_id"],
        static_reals=[],  # Example: "market_cap_static_norm"
        time_varying_known_reals=[
            "day_of_week",
            "day_of_month",
            "month",
            "is_quarter_end",
            "time_idx_norm",  # Assuming a normalized time_idx might be used
        ],
        time_varying_unknown_reals=[
            "log_return_1d",
            "log_return_5d",
            "log_return_20d",
            "volume_zscore_20d",
            "volatility_20d",
            "rsi_14d",
            "macd",
            "close_norm",  # Example: "close_norm"
        ],
        target_columns=["target_1d", "target_5d", "target_20d"],
        lookback_days=20,  # Smaller lookback for faster tests
    )


# Fixture for a tiny TimeSeriesDataSet using processed_data
# The `processed_data` fixture is defined in `tests/test_data.py`.
# Pytest should be able to inject it if run together.
@pytest.fixture(scope="module")
def tiny_timeseries_dataset(processed_data, data_config_for_tests):
    if processed_data.height == 0:
        pytest.skip("Processed data is empty, skipping TFT dataset creation.")

    # Take a small slice for faster testing
    # Ensure multiple tickers and sufficient history for lookback + prediction
    # Let's try to get at least 2 tickers with enough data
    ticker_counts = processed_data["ticker_id"].value_counts()
    valid_tickers = ticker_counts[
        ticker_counts > (data_config_for_tests.lookback_days + 5)
    ].index  # 5 for prediction horizon

    if len(valid_tickers) < 1:
        pytest.skip(
            "Not enough tickers with sufficient data for tiny_timeseries_dataset."
        )

    # Select data for a few tickers to make it tiny but representative
    # Using .head(2000) from prompt, but ensuring variety
    sample_data_pd = (
        processed_data.filter(pl.col("ticker_id").is_in(valid_tickers[:2]))
        .to_pandas()
        .head(2000)
    )

    if sample_data_pd.empty:
        pytest.skip("Sampled data for tiny_timeseries_dataset is empty.")

    # Ensure time_idx is int
    sample_data_pd["time_idx"] = sample_data_pd["time_idx"].astype(np.int64)
    # Ensure categoricals are string
    for cat_col in data_config_for_tests.static_categoricals:
        if cat_col in sample_data_pd.columns:
            sample_data_pd[cat_col] = sample_data_pd[cat_col].astype(str)

    # Minimal set of features required by TimeSeriesDataSet constructor based on DataConfig
    required_cols_for_tft = (
        data_config_for_tests.static_categoricals
        + data_config_for_tests.static_reals
        + data_config_for_tests.time_varying_known_reals
        + data_config_for_tests.time_varying_unknown_reals
        + data_config_for_tests.target_columns
        + ["time_idx", "ticker_id"]  # 'ticker_id' assumed as group_id
    )

    # Create dummy columns if they are in config but not in processed_data (e.g. _norm versions)
    for col in required_cols_for_tft:
        if (
            col not in sample_data_pd.columns
            and col not in data_config_for_tests.static_categoricals
        ):  # Static cats are handled
            if (
                col in data_config_for_tests.static_reals
                or col in data_config_for_tests.time_varying_known_reals
                or col in data_config_for_tests.time_varying_unknown_reals
            ):
                sample_data_pd[col] = 0.0  # Add dummy real column
            elif col in data_config_for_tests.target_columns:
                sample_data_pd[col] = 0.0  # Add dummy target

    # Ensure all required columns are present now
    missing_cols = [
        col
        for col in required_cols_for_tft
        if col not in sample_data_pd.columns
        and col not in data_config_for_tests.static_categoricals + ["ticker_id"]
    ]  # ticker_id is group
    if missing_cols:
        pytest.skip(
            f"Missing columns for TimeSeriesDataSet: {missing_cols} from {sample_data_pd.columns}"
        )

    categorical_encoders = {}
    for cat_col in data_config_for_tests.static_categoricals:
        if cat_col in sample_data_pd.columns and sample_data_pd[cat_col].isnull().any():
            categorical_encoders[cat_col] = NaNLabelEncoder(add_nan=True)

    ds = TimeSeriesDataSet(
        sample_data_pd,
        time_idx="time_idx",
        target=data_config_for_tests.target_columns[
            0
        ],  # Single target for simplicity in test
        group_ids=["ticker_id"],  # Assuming 'ticker_id' is the group
        static_categoricals=data_config_for_tests.static_categoricals,
        static_reals=[
            col
            for col in data_config_for_tests.static_reals
            if col in sample_data_pd.columns
        ],
        time_varying_known_categoricals=[],  # Add if any
        time_varying_known_reals=[
            col
            for col in data_config_for_tests.time_varying_known_reals
            if col in sample_data_pd.columns
        ],
        time_varying_unknown_categoricals=[],  # Add if any
        time_varying_unknown_reals=[
            col
            for col in data_config_for_tests.time_varying_unknown_reals
            if col in sample_data_pd.columns
        ],
        max_encoder_length=data_config_for_tests.lookback_days,
        max_prediction_length=1,  # Predict 1 step ahead for test
        categorical_encoders=categorical_encoders if categorical_encoders else {},
        add_relative_time_idx=True,
        allow_missing_timesteps=True,  # Important
    )
    return ds


def test_tft_forward(tiny_timeseries_dataset):
    if tiny_timeseries_dataset is None:
        pytest.skip("Tiny TimeSeriesDataSet could not be created.")

    # Calculate embedding_sizes based on the tiny_timeseries_dataset
    embedding_sizes_calc = {}
    if tiny_timeseries_dataset.static_categoricals:
        for cat_col in tiny_timeseries_dataset.static_categoricals:
            # Assuming NaNLabelEncoder or similar is used, which has 'cardinality'
            vocab_size = tiny_timeseries_dataset.categorical_encoders[
                cat_col
            ].cardinality
            dim = min(round(vocab_size**0.25), 32)
            embedding_sizes_calc[cat_col] = int(dim) if dim > 0 else 1

    # Minimal model_specific_params, embedding_sizes will be added
    model_params = {
        "hidden_size": 16,  # Small hidden size for test
        "lstm_layers": 1,  # Small LSTM layers
        "dropout": 0.1,
        "output_size": 3,  # For quantiles [0.05, 0.5, 0.95] if loss is QuantileLoss
        # Or len(targets) if not using quantiles in this specific test model instance
        "embedding_sizes": embedding_sizes_calc,  # Pass calculated embedding sizes
    }

    # If QuantileLoss is used, define it
    from pytorch_forecasting.metrics import QuantileLoss

    model_params["loss"] = QuantileLoss(quantiles=[0.05, 0.5, 0.95])

    model = GlobalTFT(
        timeseries_dataset=tiny_timeseries_dataset,
        model_specific_params=model_params,
        learning_rate=1e-3,
    )

    # Create a dataloader for the test
    # Use a small batch size
    dataloader = tiny_timeseries_dataset.to_dataloader(
        train=True, batch_size=4, shuffle=False
    )  # train=True to get some data

    try:
        batch = next(iter(dataloader))
    except StopIteration:
        pytest.skip("Tiny dataset produced an empty dataloader.")
        return

    x, y = batch
    out = model(x)  # x is the input dictionary from the dataloader

    # Check output shape: (batch_size, max_prediction_length, output_size)
    # output_size depends on if quantiles are used (e.g. 3 for 3 quantiles, or 1 if single point forecast * num_targets)
    # For QuantileLoss with [0.05, 0.5, 0.95], output_size is 3 (if single target)
    # If multiple targets, it's len(targets) * num_quantiles typically
    # The TFT `output_size` param controls the last dimension. If default quantiles for loss, it's num_quantiles.
    # The default target is single (first from list).

    assert out.prediction.shape[0] == 4  # batch size
    # out.prediction.shape[1] is max_prediction_length
    # out.prediction.shape[2] is output_size (number of quantiles if QuantileLoss)
    assert out.prediction.shape[1] == tiny_timeseries_dataset.max_prediction_length

    # Check if output_size matches number of quantiles if QuantileLoss is used.
    if isinstance(model.model.loss, QuantileLoss):
        assert out.prediction.shape[2] == len(model.model.loss.quantiles)
    else:  # Or if some other loss, it might be just 1 (for point forecast) per target
        assert out.prediction.shape[2] == model.hparams.model_specific_params.get(
            "output_size", 1
        )

    assert torch.isfinite(out.prediction).all()


def test_balanced_sampler(processed_data):
    if processed_data.empty:
        pytest.skip("Processed data is empty, skipping balanced sampler test.")

    # Use a subset of processed_data to speed up test if it's very large
    # but ensure enough variety. The prompt implies using the full processed_data.
    data_pd_for_sampler = (
        processed_data.to_pandas()
    )  # Convert polars to pandas for value_counts/reindex

    # Ensure 'ticker_id' is present
    if "ticker_id" not in data_pd_for_sampler.columns:
        pytest.skip("Column 'ticker_id' not found in processed_data.")

    counts = data_pd_for_sampler["ticker_id"].value_counts()

    # If only one ticker, ratio test is meaningless
    if len(counts) < 2:
        pytest.skip("Balanced sampler test requires at least 2 unique tickers.")

    # Weights for WeightedRandomSampler: 1 / count of ticker for each sample
    # The `weights` Series should have the same index as `data_pd_for_sampler`
    # and values should be 1/count_of_ticker_for_that_row
    ticker_to_weight = 1.0 / counts
    row_weights = (
        data_pd_for_sampler["ticker_id"].map(ticker_to_weight).fillna(1.0).values
    )  # .values to get numpy array

    num_samples_to_draw = min(
        1000, len(data_pd_for_sampler)
    )  # Draw 1000 samples or dataset size

    sampler = WeightedRandomSampler(
        weights=row_weights, num_samples=num_samples_to_draw, replacement=True
    )

    # Get the ticker_ids of the sampled indices
    sampled_indices = list(sampler)
    sampled_ids = data_pd_for_sampler["ticker_id"].iloc[sampled_indices]

    ratio = sampled_ids.value_counts(normalize=True)

    # Assert that high-volume tickers do not excessively dominate
    # The condition ratio.max() / ratio.min() < 5
    # This test can be sensitive if some tickers have very few samples in `processed_data`
    # or if num_samples_to_draw is small relative to number of tickers.
    # For robustness, consider filtering tickers with very low counts before this test,
    # or adjusting the num_samples_to_draw and the threshold.

    min_ticker_count_for_ratio_test = (
        5  # Tickers must appear at least this many times in the sample
    )
    valid_ratios = ratio[ratio * num_samples_to_draw >= min_ticker_count_for_ratio_test]

    if len(valid_ratios) < 2:
        print(
            f"Warning: Balanced sampler test could not be robustly performed. Sampled ratios: {ratio}. Valid ratios for test: {valid_ratios}"
        )
        # Potentially skip or assert True if not enough variety in sample
        # For now, proceed with original assert but be mindful it might fail on sparse data.
        if ratio.empty:  # if no samples drawn or no valid ratios
            assert True  # or skip
            return

    # print(f"Sampler test:counts:\n{counts}")
    # print(f"Sampler test: ratios:\n{ratio}")
    # print(f"Sampler test: ratio.max() = {ratio.max()}, ratio.min() = {ratio.min()}")
    # print(f"Sampler test: ratio.max() / ratio.min() = {ratio.max() / ratio.min()}")

    assert (
        ratio.max() / ratio.min()
    ) < 5, f"Sampling imbalance detected: max/min ratio is {ratio.max() / ratio.min()}"


===== conf/config.yaml =====
# conf/config.yaml
defaults:
  - data: default_data_config # We'll create this file next
  - model: tft_default       # And this one
  - trainer: default_trainer # And this
  - _self_ # Allows overriding defaults from this file itself

# General experiment parameters
project_name: "market_prediction_workbench"
experiment_id: "B_default" # Can be overridden by CLI or specific experiment configs
seed: 42 # For reproducibility

# Paths (can be relative to project root or absolute if needed)
paths:
  data_dir: "data"
  raw_data_file: "${paths.data_dir}/raw/stock_data.csv"
  processed_data_file: "${paths.data_dir}/processed/processed_data.parquet"
  log_dir: "experiments" # For PyTorch Lightning logs, MLflow, etc.
  # ticker_map_file: "${paths.data_dir}/processed/ticker_map.parquet" # Might be needed for num_tickers
  # sector_map_file: "${paths.data_dir}/processed/sector_map.parquet" # Might be needed for num_sectors

evaluate:
  batch_size: 4096
  num_workers: 8
  sample_tickers: ["AAPL", "MSFT", "GE"]  # For visualization


===== conf/trainer/default_trainer.yaml =====
# conf/trainer/default_trainer.yaml
epochs: 10 # Default number of epochs
batch_size: 4096 # Default batch size for training
num_workers: 8 # Default num_workers for DataLoaders, adjust based on system
# patience_for_early_stopping: 5

# PyTorch Lightning Trainer arguments
# These can be overridden from command line e.g. trainer.max_epochs=1
max_epochs: ${trainer.epochs}
# gpus: 1 # or devices=1, accelerator="gpu"
accelerator: "auto" # PTL will try to pick best available
devices: "auto" # "auto", 1 (for 1 GPU), or [0,1] for specific GPUs

# For EarlyStopping
early_stopping_monitor: "val_loss"
early_stopping_patience: 5
early_stopping_mode: "min"

# For LearningRateMonitor
lr_monitor_logging_interval: "epoch"

# gradient clipping to prevent explosions:
gradient_clip_val: 1.0

# WandB Logger (optional)
use_wandb: true # <--- CHANGE THIS TO true
wandb_project_name: "market-tft" # You can change this if you like
wandb_entity: "rolandpolczer-roland-polczer" # <--- REPLACE THIS with your W&B username or team name


===== conf/model/tft_default.yaml =====
# conf/model/tft_default.yaml
_target_: market_prediction_workbench.model.GlobalTFT # Changed this line

# Parameters for GlobalTFT constructor
learning_rate: 1e-3 # Added
weight_decay: 1e-5  # Added

# Hyperparameters for the underlying TemporalFusionTransformer
# These will be filtered into model_specific_params in train.py
hidden_size: 64
lstm_layers: 2
dropout: 0.10
loss:
  _target_: pytorch_forecasting.metrics.QuantileLoss
  quantiles: [0.05, 0.5, 0.95]
# embedding_sizes will be calculated in train.py and added to model_specific_params
