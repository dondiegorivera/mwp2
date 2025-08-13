# scripts/audit_outliers.py
# Usage:
#   poetry run python scripts/audit_outliers.py \
#       --run-dir experiments/evaluation/<RUN_ID> \
#       --data-parquet data/processed/processed_data.parquet \
#       --ticker-map data/processed/ticker_map.parquet \
#       --topk 200
#
# Produces:
#   <run-dir>/audit_summary.json
#   <run-dir>/outliers_1d.csv, outliers_5d.csv, outliers_20d.csv
#   <run-dir>/per_ticker_1d.csv, per_ticker_5d.csv, per_ticker_20d.csv
#   (optional) <run-dir>/resid_hist_*.png

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def infer_targets(preds_df: pd.DataFrame) -> list[str]:
    """
    Targets are columns like '<name>@h1' (not *_lower/_upper/_cal).
    In your repo these are: '1d','5d','20d'.
    """
    names = set()
    for c in preds_df.columns:
        if "@h1" in c and ("_lower" not in c and "_upper" not in c):
            names.add(c.split("@h1")[0])
    return sorted(names)

def load_data(run_dir: Path, data_parquet: Path, ticker_map_path: Path | None):
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.csv not found at {pred_path}")
    preds = pd.read_csv(pred_path)
    # normalise types
    if "ticker" in preds.columns:
        preds["ticker"] = preds["ticker"].astype(str)
    if "time_idx_h1" in preds.columns:
        preds["time_idx_h1"] = preds["time_idx_h1"].astype("int64")

    proc = pd.read_parquet(data_parquet)
    # keep only what's needed
    keep = ["ticker_id", "time_idx", "date", "target_1d", "target_5d", "target_20d"]
    keep = [c for c in keep if c in proc.columns]
    proc = proc[keep].copy()
    if "ticker_id" in proc.columns:
        proc["ticker_id"] = proc["ticker_id"].astype(str)
    if "time_idx" in proc.columns:
        proc["time_idx"] = proc["time_idx"].astype("int64")

    # add ticker_symbol if missing, using ticker_map.parquet
    if "ticker_symbol" not in preds.columns and ticker_map_path and Path(ticker_map_path).exists():
        tmap = pd.read_parquet(ticker_map_path)
        if set(["ticker_id","ticker"]).issubset(tmap.columns):
            tmap = tmap[["ticker_id","ticker"]].copy()
            tmap["ticker_id"] = tmap["ticker_id"].astype(str)
            preds = preds.merge(tmap, left_on="ticker", right_on="ticker_id", how="left")
            preds.rename(columns={"ticker":"ticker_symbol_x","ticker_x":"ticker_symbol"}, inplace=True)
            # clean columns
            if "ticker_symbol" not in preds:
                preds["ticker_symbol"] = preds.get("ticker_symbol_x", None)
            drop_cols = [c for c in ["ticker_symbol_x","ticker_id_y","ticker_y"] if c in preds.columns]
            preds.drop(columns=drop_cols, inplace=True, errors="ignore")

    return preds, proc

def join_truth(preds: pd.DataFrame, proc: pd.DataFrame) -> pd.DataFrame:
    # merge by (ticker, time_idx_h1) <-> (ticker_id, time_idx)
    mm = preds.merge(
        proc,
        left_on=["ticker", "time_idx_h1"],
        right_on=["ticker_id", "time_idx"],
        how="left",
        suffixes=("","_proc")
    )
    return mm

def compute_metrics(df: pd.DataFrame, name: str) -> dict:
    y = df[f"target_{name}"].values
    yhat = df[f"{name}@h1"].values
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() == 0:
        return {"n":0, "mae":np.nan, "rmse":np.nan, "p99_abs_resid":np.nan,
                "coverage_90":np.nan, "coverage_90_cal":np.nan}
    resid = yhat[mask] - y[mask]
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    p99 = float(np.percentile(np.abs(resid), 99))

    cov = np.nan
    cov_cal = np.nan
    lo = df.get(f"{name}_lower@h1")
    hi = df.get(f"{name}_upper@h1")
    if lo is not None and hi is not None:
        lo, hi = lo.values[mask], hi.values[mask]
        cov = float(np.mean((yhat[mask] >= lo) & (yhat[mask] <= hi)))
    lo_c = df.get(f"{name}_lower_cal@h1")
    hi_c = df.get(f"{name}_upper_cal@h1")
    if lo_c is not None and hi_c is not None:
        lo_c, hi_c = lo_c.values[mask], hi_c.values[mask]
        cov_cal = float(np.mean((yhat[mask] >= lo_c) & (yhat[mask] <= hi_c)))

    return {"n": int(mask.sum()), "mae": mae, "rmse": rmse,
            "p99_abs_resid": p99, "coverage_90": cov, "coverage_90_cal": cov_cal}

def write_outliers(df: pd.DataFrame, name: str, out_dir: Path, topk: int):
    # compute residuals
    df = df.copy()
    df[f"{name}_resid"] = df[f"{name}@h1"] - df[f"target_{name}"]
    df[f"{name}_abs_resid"] = df[f"{name}_resid"].abs()
    cols = ["ticker_symbol","ticker","time_idx_h1","date"] if "date" in df.columns else ["ticker_symbol","ticker","time_idx_h1"]
    cols = [c for c in cols if c in df.columns]
    cols += [f"target_{name}", f"{name}@h1",
             f"{name}_lower@h1", f"{name}_upper@h1",
             f"{name}_lower_cal@h1", f"{name}_upper_cal@h1",
             f"{name}_resid", f"{name}_abs_resid"]
    cols = [c for c in cols if c in df.columns]
    out = df.sort_values(f"{name}_abs_resid", ascending=False).head(topk)[cols]
    out.to_csv(out_dir / f"outliers_{name}.csv", index=False)

    # per-ticker summary
    g = df.groupby("ticker_symbol" if "ticker_symbol" in df.columns else "ticker")[f"{name}_resid"]
    per_ticker = pd.DataFrame({
        "n": g.size(),
        "mae": g.apply(lambda s: s.abs().mean()),
        "rmse": g.apply(lambda s: np.sqrt(np.mean(s**2)))
    }).sort_values("mae")
    per_ticker.to_csv(out_dir / f"per_ticker_{name}.csv")

def plot_hist(df: pd.DataFrame, name: str, out_dir: Path):
    if not {"target_"+name, f"{name}@h1"}.issubset(df.columns):
        return
    resid = (df[f"{name}@h1"] - df[f"target_{name}"]).dropna().values
    if resid.size == 0:
        return
    plt.figure(figsize=(6,4))
    plt.hist(resid, bins=200)
    plt.title(f"Residuals histogram – {name}")
    plt.xlabel("yhat - y")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"resid_hist_{name}.png")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Audit top residual outliers from predictions.csv")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--data-parquet", required=True, type=Path)
    ap.add_argument("--ticker-map", default=None, type=Path)
    ap.add_argument("--topk", type=int, default=200)
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    out_dir = run_dir  # write next to predictions.csv

    preds, proc = load_data(run_dir, args.data_parquet, args.ticker_map)
    names = infer_targets(preds)
    if not names:
        raise RuntimeError("Could not infer targets (columns like '<name>@h1') in predictions.csv")

    df = join_truth(preds, proc)

    summary = {}
    for name in names:
        # map target names like '1d' -> 'target_1d' present?
        tgt_col = f"target_{name}"
        if tgt_col not in df.columns:
            # try a safer mapping: strip non-digits? keep as is if missing
            raise RuntimeError(f"True column '{tgt_col}' not found in processed data.")
        metrics = compute_metrics(df, name)
        summary[name] = metrics
        write_outliers(df, name, out_dir, args.topk)
        plot_hist(df, name, out_dir)

    with open(out_dir / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Audit summary ===")
    for k, v in summary.items():
        print(k, v)
    print(f"\nWrote outlier tables & histograms to: {out_dir}")

if __name__ == "__main__":
    main()
