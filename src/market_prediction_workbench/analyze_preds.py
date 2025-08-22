# analyze_preds.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

# --- Inputs (edit if needed) ---
EVAL_ROOT = Path("../../experiments/evaluation")
PROCESSED_PARQUET = Path("../../data/processed/processed_data.parquet")
TICKER_MAP = Path("../../data/processed/ticker_map.parquet")  # optional


# auto-pick the latest predictions.csv under experiments/evaluation/*
def _latest_predictions_csv():
    cands = list(EVAL_ROOT.glob("*/predictions.csv"))
    if not cands:
        raise FileNotFoundError(f"No predictions.csv under {EVAL_ROOT}")
    return max(cands, key=lambda p: p.stat().st_mtime)


preds_csv = _latest_predictions_csv()
print(f"Using predictions: {preds_csv}")

# --- Load predictions & ground truth ---
preds = pd.read_csv(preds_csv)

# Ensure expected columns exist
required_pred_cols = {"ticker", "time_idx_h1"}
if not required_pred_cols.issubset(preds.columns):
    raise ValueError(
        f"predictions.csv missing {required_pred_cols - set(preds.columns)}"
    )

# If your evaluate.py already wrote true columns, we’ll use them; otherwise we’ll join from parquet
have_true_cols = any(c.endswith("@h1_true") for c in preds.columns)

if not have_true_cols:
    print(
        "True columns not found in predictions.csv – joining from processed parquet..."
    )
    base = pd.read_parquet(
        PROCESSED_PARQUET,
        columns=["ticker_id", "time_idx", "target_1d", "target_5d", "target_20d"],
    )
    base = base.rename(columns={"ticker_id": "ticker", "time_idx": "time_idx_h1"})
    df = preds.merge(base, on=["ticker", "time_idx_h1"], how="left")
    # Make explicit true columns to avoid any ambiguity later
    df["1d@h1_true"] = df["target_1d"]
    df["5d@h1_true"] = df["target_5d"]
    df["20d@h1_true"] = df["target_20d"]
else:
    df = preds.copy()

# Optional: map ticker ids to symbols if available
if TICKER_MAP.exists():
    tmap = pd.read_parquet(TICKER_MAP)  # columns: ticker, ticker_id
    df = df.merge(tmap.rename(columns={"ticker_id": "ticker"}), on="ticker", how="left")


# --- Helper to compute metrics for a single target name (e.g., "20d") ---
def per_target_analysis(df, tgt_name="20d"):
    # Column names used by evaluate.py for h1
    p50 = f"{tgt_name}@h1"
    p05 = f"{tgt_name}_lower@h1"
    p95 = f"{tgt_name}_upper@h1"
    y = f"{tgt_name}@h1_true"

    if not {p50, p05, p95, y}.issubset(df.columns):
        missing = {p50, p05, p95, y} - set(df.columns)
        raise ValueError(f"Missing columns for analysis: {missing}")

    d = df[
        (
            ["ticker", "time_idx_h1", p50, p05, p95, y, "ticker_x", "ticker_y"]
            if "ticker_x" in df.columns or "ticker_y" in df.columns
            else ["ticker", "time_idx_h1", p50, p05, p95, y]
        )
    ].copy()

    # prefer symbol column if available
    if "ticker_y" in d.columns:
        d["ticker_symbol"] = d["ticker_y"]
    elif "ticker_x" in d.columns:
        d["ticker_symbol"] = d["ticker_x"]
    else:
        d["ticker_symbol"] = d["ticker"].astype(str)

    d["resid"] = d[y] - d[p50]
    d["ae"] = d["resid"].abs()
    d["inside"] = (d[y] >= d[p05]) & (d[y] <= d[p95])

    # s_hat from your bands (approx normal): (p95 - p05) / (2 * 1.645)
    d["s_hat"] = (d[p95] - d[p05]) / (2 * 1.645)
    d["k_ratio"] = d["ae"] / (
        1.645 * d["s_hat"].replace(0, np.nan)
    )  # used for calibration

    # Overall metrics (should roughly match metrics.json)
    overall = {
        "mae": float(d["ae"].mean()),
        "rmse": float(np.sqrt((d["resid"] ** 2).mean())),
        "coverage_90": float(d["inside"].mean()),
        # simple robust scale check
        "median_s_hat": float(d["s_hat"].median(skipna=True)),
    }

    # Per-ticker breakdown
    by_ticker = (
        d.groupby(["ticker", "ticker_symbol"], dropna=False)
        .agg(
            n=("ae", "size"),
            mae=("ae", "mean"),
            rmse=("resid", lambda x: np.sqrt((x**2).mean())),
            cover_90=("inside", "mean"),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )

    # Per-date (time_idx) breakdown (averaging across tickers that day)
    by_date = (
        d.groupby("time_idx_h1")
        .agg(
            n=("ae", "size"),
            mae=("ae", "mean"),
            rmse=("resid", lambda x: np.sqrt((x**2).mean())),
            cover_90=("inside", "mean"),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )

    # Simple one-parameter quantile calibration suggestion (alpha)
    # alpha = 90th percentile of k_ratio (|r| / (1.645 * s_hat))
    # Use finite values only
    finite_k = d["k_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    alpha = float(np.nanpercentile(finite_k.values, 90)) if len(finite_k) else np.nan

    return d, overall, by_ticker, by_date, alpha


def infer_target_shortnames(columns):
    names = set()
    for c in columns:
        if "@h1" in c and ("_lower" not in c and "_upper" not in c and "_cal" not in c):
            names.add(c.split("@h1")[0])
    return sorted(names)

# --- Run analyses for present targets and dump CSVs next to predictions.csv ---
OUT_DIR = preds_csv.parent
summaries = {}
present_targets = infer_target_shortnames(preds.columns)
if not present_targets:
    present_targets = ["5d"]  # sensible default

for tgt in present_targets:
    print(f"\nAnalyzing {tgt} …")
    d, overall, by_ticker, by_date, alpha = per_target_analysis(df, tgt_name=tgt)
    summaries[tgt] = {"overall": overall, "alpha_for_90pc": alpha}

    # Save breakdowns
    by_ticker.to_csv(OUT_DIR / f"breakdown_{tgt}_by_ticker.csv", index=False)
    by_date.to_csv(OUT_DIR / f"breakdown_{tgt}_by_date.csv", index=False)

    # Print quick view
    print("Overall:", overall)
    print(f"Suggested α for 90% coverage calibration (apply to s_hat): {alpha:.3f}")
    print("\nWorst tickers by MAE (top 10):")
    print(by_ticker.head(10).to_string(index=False))
    print("\nWorst dates by MAE (top 10):")
    print(by_date.head(10).to_string(index=False))

# Also write a compact JSON summary
with open(OUT_DIR / "analysis_summary.json", "w") as f:
    json.dump(summaries, f, indent=2)

print(f"\nSaved per-ticker/per-date breakdowns and summary to: {OUT_DIR}")
