# scripts/filter_universe.py
# Usage:
# poetry run python scripts/filter_universe.py \
#   --in data/processed/processed_data.cleaned.parquet \
#   --ticker-map data/processed/ticker_map.parquet \
#   --out data/processed/processed_data.eqonly.parquet \
#   --auto-target-tickers 1000 --min-price 3 --min-history-days 500 \
#   --price-lookback-days 60 --liq-lookback-days 120 --min-nonzero-volume-frac 0.05 \
#   --drop-event-windows --event-threshold 0.25 --event-window 2

import argparse
from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
import re
from typing import Tuple

RX_NON_EQ = [
    r".*=X$",  # FX pairs, e.g. EURUSD=X
    r".*=F$",  # Futures, e.g. CL=F
    r"^[A-Z0-9\-]+-USD$",  # crypto pairs
    r"^[A-Z0-9\-]+-USDT$",
    r"^[A-Z0-9\-]+-EUR$",
    r"^BTC.*|^ETH.*|^DOGE.*|^SHIB.*",
    r"^\^.*",  # indices (^GSPC)
    r".*\.WS$|.*\.W$|.*-WS$",
]


def _bool_series(df, init=False):
    return pd.Series(np.full(len(df), init, dtype=bool), index=df.index)


def build_exclusion_mask(sym: pd.Series) -> pd.Series:
    bad = _bool_series(sym, init=False)
    for rx in RX_NON_EQ:
        bad |= sym.str.contains(rx, regex=True, na=False)
    for pat in ["=F", "-USDT", "-USD", "-EUR"]:
        bad |= sym.str.contains(re.escape(pat), regex=True, na=False)
    return bad


def recent_median_price_flag(
    df: pd.DataFrame, price_col: str, lookback_days: int, min_price: float
) -> pd.Series:
    # median price over the last N trading days (per ticker)
    last_idx = df.groupby("ticker_id")["time_idx"].transform("max")
    is_recent = df["time_idx"] >= (last_idx - lookback_days + 1)
    recent = df.loc[
        is_recent & (df["is_missing"] < 0.5), ["ticker_id", price_col]
    ].copy()
    med = recent.groupby("ticker_id")[price_col].median().rename("recent_median_price")
    df = df.merge(med, on="ticker_id", how="left")
    return df["recent_median_price"].fillna(0) < min_price


def compute_liquidity_flags(
    df: pd.DataFrame,
    price_col: str,
    liq_lookback_days: int,
    min_nonzero_frac: float,
    auto_target_tickers: int,
    min_value20: float,
):
    df = df.sort_values(["ticker_id", "time_idx"]).copy()
    # value traded
    price = pd.to_numeric(df[price_col], errors="coerce")
    vol = pd.to_numeric(df.get("volume", np.nan), errors="coerce")
    df["value_traded"] = price * vol

    # rolling 20d median per ticker
    df["value_med20"] = df.groupby("ticker_id", sort=False)["value_traded"].transform(
        lambda s: s.rolling(20, min_periods=10).median()
    )

    # restrict to last liq_lookback_days
    last_idx = df.groupby("ticker_id")["time_idx"].transform("max")
    recent_mask = df["time_idx"] >= (last_idx - liq_lookback_days + 1)
    recent = df.loc[recent_mask].copy()

    # per-ticker recent characteristic
    med20_recent = (
        recent.groupby("ticker_id")["value_med20"]
        .median()
        .rename("median_value20_recent")
    )

    # fraction of recent days with nonzero volume (fallback)
    nz_frac = (
        (recent["value_traded"] > 0)
        .groupby(recent["ticker_id"])
        .mean()
        .rename("nonzero_volume_frac")
    )

    liq = pd.concat([med20_recent, nz_frac], axis=1)

    # pick threshold
    if auto_target_tickers and auto_target_tickers > 0:
        pos = liq["median_value20_recent"].replace([np.inf, -np.inf], np.nan)
        pos = pos[pos > 0].dropna()
        if len(pos) > 0:
            # k-th largest among positives
            k = min(auto_target_tickers - 1, len(pos) - 1)
            cutoff = pos.sort_values(ascending=False).iloc[k]
            # fallback: if still tiny, use 25th pctile of positives or provided absolute min
            if cutoff <= 0:
                cutoff = max(float(np.percentile(pos, 25)), float(min_value20))
        else:
            cutoff = float(min_value20)
        print(
            f"[auto] median_value20_recent cutoff (top-{auto_target_tickers}): {cutoff:,.0f}"
        )
        min_value20_eff = float(cutoff)
    else:
        min_value20_eff = float(min_value20)

    # build flags (ticker-level)
    liq["liq_value_bad"] = liq["median_value20_recent"].fillna(0) < min_value20_eff
    liq["liq_nzfrac_bad"] = liq["nonzero_volume_frac"].fillna(0) < float(
        min_nonzero_frac
    )

    return df.merge(liq, on="ticker_id", how="left"), liq


def compute_event_windows_from_raw_prices(
    df: pd.DataFrame, price_col: str, threshold: float, window: int
) -> Tuple[pd.DataFrame, int]:
    """
    Compute trading-day log return from raw prices (is_missing == 0) and drop ±window days
    around large |ret| >= threshold.
    """
    d = df.sort_values(["ticker_id", "time_idx"]).copy()
    # compute trading-day log return (ignore calendar gaps)
    # mark trading-only rows
    is_trade = d["is_missing"] < 0.5
    d["log_px"] = np.log(pd.to_numeric(d[price_col], errors="coerce").clip(lower=1e-6))
    # per ticker shift only on trading rows
    d["log_px_td"] = d["log_px"].where(is_trade)
    d["ret_1d_raw"] = d.groupby("ticker_id")["log_px_td"].diff()

    to_drop = _bool_series(d, init=False)
    for tid, g in d.groupby("ticker_id", sort=False):
        if g.empty:
            continue
        big = g["ret_1d_raw"].abs() >= threshold
        if not big.any():
            continue
        idxs = g.loc[big, "time_idx"].to_numpy()
        lo = idxs - window
        hi = idxs + window
        gi = g["time_idx"].to_numpy()
        mask = np.zeros(len(g), dtype=bool)
        for a, b in zip(lo, hi):
            mask |= (gi >= a) & (gi <= b)
        to_drop.loc[g.index] = mask

    dropped = int(to_drop.sum())
    return d.loc[~to_drop].copy(), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--ticker-map", dest="tickermap", required=True, type=Path)
    ap.add_argument("--out", dest="out", required=True, type=Path)

    ap.add_argument("--min-price", type=float, default=3.0)
    ap.add_argument("--price-lookback-days", type=int, default=60)
    ap.add_argument("--min-history-days", type=int, default=500)

    ap.add_argument("--liq-lookback-days", type=int, default=120)
    ap.add_argument("--min-median-value20", type=float, default=1_000_000.0)
    ap.add_argument("--min-nonzero-volume-frac", type=float, default=0.05)
    ap.add_argument("--auto-target-tickers", type=int, default=0)

    ap.add_argument(
        "--drop-event-windows", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--event-threshold", type=float, default=0.25)
    ap.add_argument("--event-window", type=int, default=2)

    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    tmap = pl.read_parquet(args.tickermap)
    id2sym = dict(
        zip(tmap["ticker_id"].cast(pl.Utf8).to_list(), tmap["ticker"].to_list())
    )
    df["ticker_symbol"] = df["ticker_id"].astype(str).map(id2sym).fillna("UNK")

    price_col = "adj_close" if "adj_close" in df.columns else "close"

    total_rows = len(df)
    uniq = df["ticker_id"].nunique()
    print(f"Input rows: {total_rows:,} | tickers: {uniq:,}")

    # --- hygiene: non-equities ---
    sym = df["ticker_symbol"].astype(str)
    non_eq = build_exclusion_mask(sym)

    # --- min history (trading rows only) ---
    non_miss = (
        pd.to_numeric(df.get("is_missing", 0.0), errors="coerce").fillna(1.0) < 0.5
    )
    hist_counts = (
        df.loc[non_miss]
        .groupby("ticker_id")["time_idx"]
        .size()
        .rename("hist_non_missing")
    )
    df = df.merge(hist_counts, on="ticker_id", how="left")
    hist_bad = df["hist_non_missing"].fillna(0) < int(args.min_history_days)

    # --- recent price floor (median over last N trading days) ---
    price_bad = recent_median_price_flag(
        df, price_col, args.price_lookback_days, float(args.min_price)
    )

    # --- liquidity (recent window) & fallback for no-volume feeds ---
    df, liq = compute_liquidity_flags(
        df,
        price_col,
        args.liq_lookback_days,
        float(args.min_nonzero_volume_frac),
        int(args.auto_target_tickers),
        float(args.min_median_value20),
    )
    # liq flags per ticker
    # final liquidity flag = both value and nonzero-volume checks failing OR either failing
    # (choose stricter: require both to pass)
    liq_bad = (liq["liq_value_bad"] | liq["liq_nzfrac_bad"]).rename("liq_bad")
    df = df.merge(liq_bad, on="ticker_id", how="left")

    # --- combine ticker-level flags ---
    per_ticker_bad = (
        df.assign(
            non_eq=non_eq,
            price_bad=price_bad,
            hist_bad=hist_bad,
            liquid_bad=df["liq_bad"].fillna(True),  # missing -> bad
        )
        .groupby("ticker_id")[["non_eq", "price_bad", "hist_bad", "liquid_bad"]]
        .max()
    )

    def _examples(mask_name):
        tids = per_ticker_bad.index[per_ticker_bad[mask_name]]
        syms = [id2sym.get(str(t), "UNK") for t in map(str, tids[:20])]
        return syms

    print("\n=== Exclusion (ticker-level) ===")
    for name in ["non_eq", "price_bad", "hist_bad", "liquid_bad"]:
        n = int(per_ticker_bad[name].sum())
        print(f"{name:>10}: {n:>6} tickers | examples: { _examples(name) }")

    keep_tickers = set(per_ticker_bad.index[~per_ticker_bad.any(axis=1)])
    df_kept = df[df["ticker_id"].isin(keep_tickers)].copy()

    # --- event landmines from raw prices (works even after cleaning targets) ---
    dropped_events = 0
    if args.drop_event_windows:
        before = len(df_kept)
        df_kept, dropped_events = compute_event_windows_from_raw_prices(
            df_kept,
            price_col=price_col,
            threshold=float(args.event_threshold),
            window=int(args.event_window),
        )
        after = len(df_kept)
        print(
            f"\nEvent windows dropped: rows {before:,} -> {after:,} (−{before - after:,}) "
            f"using |raw 1d logret| ≥ {args.event_threshold} & window ±{args.event_window}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = df_kept.drop(columns=["hist_non_missing"], errors="ignore")
    out.to_parquet(args.out, index=False)

    kept_rows = len(out)
    kept_tickers = out["ticker_id"].nunique()
    print(f"\nOutput rows: {kept_rows:,} | tickers: {kept_tickers:,}")
    all_bad_tids = per_ticker_bad.index[per_ticker_bad.any(axis=1)]
    examples = [id2sym.get(str(t), "UNK") for t in map(str, all_bad_tids[:20])]
    print("Examples removed:", examples)


if __name__ == "__main__":
    main()
