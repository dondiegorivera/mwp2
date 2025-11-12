# src/market_prediction_workbench/data.py
import polars as pl
from pathlib import Path
import numpy as _np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
import math


@dataclass
class DataConfig:
    static_reals: list[str]
    static_categoricals: list[str]
    time_varying_known_reals: list[str]
    time_varying_unknown_reals: list[str]
    target_columns: list[str]
    lookback_days: int = 120
    prediction_horizon: int = 1


class MarketDataset(Dataset):
    def __init__(self, data: pl.DataFrame, config: DataConfig):
        super().__init__()
        self.config = config
        self.data = data.with_row_index(name="_original_idx_for_dataset")

        group_sizes = self.data.group_by("ticker_id").len().rename({"len": "size"})
        data_with_size_and_orig_idx = self.data.join(group_sizes, on="ticker_id")

        max_horizon = 20
        valid_rows_df = data_with_size_and_orig_idx.filter(
            (pl.col("time_idx") >= (config.lookback_days - 1))
            & (pl.col("time_idx") < (pl.col("size") - max_horizon))
        )

        self.valid_indices = (
            valid_rows_df.select("_original_idx_for_dataset").to_series().to_list()
        )
        print(f"Created dataset with {len(self.valid_indices)} valid samples.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_df_idx = self.valid_indices[idx]
        end_row_data = self.data[original_df_idx]

        ticker_id = end_row_data.select("ticker_id").item()
        end_time_idx_for_sample = end_row_data.select("time_idx").item()
        start_time_idx_for_sample = (
            end_time_idx_for_sample - self.config.lookback_days + 1
        )

        window_df = self.data.filter(
            (pl.col("ticker_id") == ticker_id)
            & (pl.col("time_idx") >= start_time_idx_for_sample)
            & (pl.col("time_idx") <= end_time_idx_for_sample)
        )

        if window_df.height != self.config.lookback_days:
            raise ValueError(
                f"Sample {idx} for ticker {ticker_id} (orig_idx {original_df_idx}): "
                f"Expected lookback window of {self.config.lookback_days} days, got {window_df.height}."
            )

        target_values_series = end_row_data.select(self.config.target_columns).row(0)

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
        y = torch.tensor(target_values_series, dtype=torch.float32)

        return {
            "x_cat": x_static_cats,
            "x_known_reals": x_known_reals,
            "x_unknown_reals": x_unknown_reals,
            "y": y,
            "groups": torch.tensor([ticker_id], dtype=torch.long),
            "time_idx_window": torch.tensor(
                window_df.select("time_idx").to_series().to_list(), dtype=torch.long
            ),
        }


def _clean_col_names(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({col: col.lower().replace(" ", "_") for col in df.columns})


def load_and_clean_data(csv_path: Path) -> pl.DataFrame:
    print(f"Loading data from {csv_path}...")
    common_null_values = ["N/A", "NA", "NaN", "", "null", "#N/A"]

    df = pl.read_csv(
        csv_path,
        try_parse_dates=True,
        null_values=common_null_values,
        infer_schema_length=10000,
    )
    df = _clean_col_names(df)

    df = df.select(
        pl.col("date"),
        pl.col("asset_ticker").alias("ticker"),
        pl.col("closing_price").alias("close"),
        pl.col("opening_price").alias("open"),
        pl.col("volume").cast(pl.Int64, strict=False),
        pl.col("industry").fill_null("N/A"),
        pl.col("market_cap").cast(pl.Float64, strict=False),
    )

    df = df.drop_nulls(subset=["date", "ticker", "close"])

    print("Initial cleaning and type casting complete.")
    print("DataFrame shape:", df.shape)
    print("Sample data:")
    print(df.head())

    return df


def create_mappings(df: pl.DataFrame, output_dir: Path) -> pl.DataFrame:
    print("Creating ticker and industry mappings...")

    ticker_map = (
        df.select("ticker").unique().sort("ticker").with_row_index(name="ticker_id")
    )
    industry_map = (
        df.select("industry").unique().sort("industry").with_row_index(name="sector_id")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ticker_map.write_parquet(output_dir / "ticker_map.parquet")
    industry_map.write_parquet(output_dir / "industry_map.parquet")
    print(f"Saved mappings to {output_dir}")

    df = df.join(ticker_map, on="ticker")
    df = df.join(industry_map, on="industry")
    return df


def reindex_and_fill_gaps(df: pl.DataFrame, max_ffill_days: int = 5) -> pl.DataFrame:
    print("Reindexing to a full calendar and forward-filling gaps...")
    df = df.sort("ticker_id", "date")

    def upsample_and_mark_missing_per_group(group_df: pl.DataFrame) -> pl.DataFrame:
        group_df_reindexed = group_df.upsample(time_column="date", every="1d")
        group_df_reindexed = group_df_reindexed.with_columns(
            is_missing=pl.col("close").is_null()
        )
        return group_df_reindexed

    df_reindexed_list = []
    for _, group_data in df.group_by("ticker_id", maintain_order=True):
        df_reindexed_list.append(upsample_and_mark_missing_per_group(group_data))
    if not df_reindexed_list:
        return df.clear().with_columns(
            pl.lit(None).cast(pl.Boolean).alias("is_missing")
        )

    df_reindexed = pl.concat(df_reindexed_list)

    fill_cols = [
        "close",
        "open",
        "volume",
        "ticker",
        "industry",
        "sector_id",
        "market_cap",
    ]

    df_filled = df_reindexed.with_columns(
        pl.col(fill_cols).forward_fill(limit=max_ffill_days).over("ticker_id")
    )
    df_filled = df_filled.drop_nulls(subset=["ticker_id"])

    return df_filled


def create_features_and_targets(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create features and targets.

    Targets:
      * Computed on each ticker's TRADING-DAY sequence only (not calendar).
      * Horizons are 1/5/20 trading days (shift on trading days).
      * Labels exist only when both t and t+h exist; otherwise remain null and are later dropped.

    Context features (daily, known at t):
      * mkt_ret_1d: cross-sectional median of log_return_1d over tickers trading on date t
      * sector_ret_1d: same median within sector_id
      * mkt_vol_20d: rolling std(20) of mkt_ret_1d (over calendar dates)
      * sector_vol_20d: rolling std(20) of sector_ret_1d per sector
      * Optional macro joins from data/external/macro.parquet: VIX, VSTOXX, US2Y, US10Y, DE2Y, DE10Y

    No clipping/winsorization here (handled later by winsorize_targets.py).
    """
    print("Creating features and targets...")
    df = df.sort("ticker_id", "date")

    # --- choose price column (prefer adjusted if present) ---
    price_col = "adj_close" if "adj_close" in df.columns else "close"
    if price_col != "adj_close":
        print(
            "Adjusted close ('adj_close') not found; using 'close' for labels/features."
        )
    safe_price = (
        pl.when(pl.col(price_col) <= 1e-6).then(1e-6).otherwise(pl.col(price_col))
    )
    log_close = safe_price.log()

    # --- Known-future (calendar) features ---
    df = df.with_columns(
        day_of_week=pl.col("date").dt.weekday().cast(pl.Float32),
        day_of_month=pl.col("date").dt.day().cast(pl.Float32),
        month=pl.col("date").dt.month().cast(pl.Float32),
        is_quarter_end=(
            (pl.col("date").dt.month().is_in([3, 6, 9, 12]))
            & (pl.col("date").dt.month_end() == pl.col("date"))
        ),
    ).with_columns(pl.col("is_quarter_end").cast(pl.Float32))

    # --- Cyclical encodings ---
    two_pi = 2.0 * math.pi
    df = df.with_columns(
        dow_sin=(pl.col("day_of_week") * (two_pi / 7.0)).sin().cast(pl.Float32),
        dow_cos=(pl.col("day_of_week") * (two_pi / 7.0)).cos().cast(pl.Float32),
        mon_sin=(pl.col("month") * (two_pi / 12.0)).sin().cast(pl.Float32),
        mon_cos=(pl.col("month") * (two_pi / 12.0)).cos().cast(pl.Float32),
    )

    # --- Ensure is_missing present (True for calendar-upsampled gaps BEFORE ffill) ---
    if "is_missing" in df.columns:
        is_missing_mask = pl.col("is_missing").cast(pl.Boolean)
        df = df.with_columns(pl.col("is_missing").cast(pl.Float32))
    else:
        print("Warning: 'is_missing' column not found before feature engineering.")
        df = df.with_columns(pl.lit(0.0).cast(pl.Float32).alias("is_missing"))
        is_missing_mask = pl.lit(False)

    # --- Observed-past features (calendar series masked for gaps) ---
    log_return_1d_base_expr = log_close - log_close.shift(1)
    rolling_volume_mean = pl.col("volume").rolling_mean(window_size=20, min_samples=10)
    rolling_volume_std = pl.col("volume").rolling_std(window_size=20, min_samples=10)
    volume_zscore_expr = (
        pl.when(rolling_volume_std > 1e-6)
        .then((pl.col("volume") - rolling_volume_mean) / rolling_volume_std)
        .otherwise(0.0)
    )

    if "market_cap" in df.columns and "ticker_id" in df.columns:
        # Defensive: non-negative, log1p, then aggregate per ticker
        df = df.with_columns(
            pl.when(pl.col("market_cap").cast(pl.Float64) < 0.0)
            .then(pl.lit(0.0))
            .otherwise(pl.col("market_cap").cast(pl.Float64))
            .fill_null(0.0)
            .alias("market_cap_clean")
        )

        # log market cap per row
        df = df.with_columns((pl.col("market_cap_clean").log1p()).alias("log_mcap_row"))

        # ticker-level static: median log(mcap) across available history
        # (median is robust to occasional spikes / stale caps)
        log_mcap_static = df.group_by("ticker_id").agg(
            pl.col("log_mcap_row").median().alias("log_mcap_static")
        )

        df = df.join(log_mcap_static, on="ticker_id", how="left")

        # global cross-sectional z-score over tickers (kept *static* per ticker)
        stats = log_mcap_static.select(
            [
                pl.col("log_mcap_static").mean().alias("m_mu"),
                pl.col("log_mcap_static").std().alias("m_sd"),
            ]
        ).row(0)
        m_mu, m_sd = (
            float(stats[0]),
            float(stats[1] if stats[1] and stats[1] > 0 else 1.0),
        )

        df = df.with_columns(
            ((pl.col("log_mcap_static") - m_mu) / m_sd).alias("market_cap_static_norm")
        )

        # Optional hygiene: drop helpers if you don't need them downstream
        df = df.drop(["market_cap_clean", "log_mcap_row"])
    else:
        print("[warn] market_cap or ticker_id missing — skipping static size feature.")

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
        price_change=(pl.when(~is_missing_mask).then(safe_price.diff(1))).over(
            "ticker_id"
        ),
    )

    # --- RSI (14) ---
    df = df.with_columns(
        gain=pl.when(pl.col("price_change") > 0)
        .then(pl.col("price_change"))
        .otherwise(0.0),
        loss=pl.when(pl.col("price_change") < 0)
        .then(-pl.col("price_change"))
        .otherwise(0.0),
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

    # --- MACD (12/26 EMA, 9-signal) ---
    ema_12_expr = safe_price.ewm_mean(span=12, min_samples=11)
    ema_26_expr = safe_price.ewm_mean(span=26, min_samples=25)
    df = df.with_columns(
        macd_base=(pl.when(~is_missing_mask).then(ema_12_expr - ema_26_expr)).over(
            "ticker_id"
        )
    )
    macd_signal_expr = pl.col("macd_base").ewm_mean(span=9, min_samples=8)
    df = df.with_columns(
        macd_signal_base=(pl.when(~is_missing_mask).then(macd_signal_expr)).over(
            "ticker_id"
        )
    )
    df = df.rename({"macd_base": "macd", "macd_signal_base": "macd_signal"})

    # --- CLEANUP feature scratch columns ---
    intermediate_cols = ["price_change", "gain", "loss", "avg_gain", "avg_loss"]
    cols_to_drop_now = [col for col in intermediate_cols if col in df.columns]
    if cols_to_drop_now:
        df = df.drop(cols_to_drop_now)

    # --- Convert NaN/inf to nulls for float columns ---
    float_cols = [cn for cn, dt in df.schema.items() if dt in [pl.Float32, pl.Float64]]
    for col_name in float_cols:
        df = df.with_columns(
            pl.when(pl.col(col_name).is_nan() | pl.col(col_name).is_infinite())
            .then(None)
            .otherwise(pl.col(col_name))
            .alias(col_name)
        )

    # ---------------------------------------------------------------------
    # TRADING-DAY TARGETS (labels) — computed ONLY on non-missing (trading) rows
    # ---------------------------------------------------------------------
    print("Computing trading-day targets (1/5/20)…")
    trading_only = (
        df.filter(~pl.col("is_missing").cast(pl.Boolean))
        .select("ticker_id", "sector_id", "date", log_close.alias("log_price_td"))
        .sort("ticker_id", "date")
    )

    trading_only = trading_only.with_columns(
        (pl.col("log_price_td").shift(-1) - pl.col("log_price_td"))
        .over("ticker_id")
        .alias("target_1d"),
        (pl.col("log_price_td").shift(-5) - pl.col("log_price_td"))
        .over("ticker_id")
        .alias("target_5d"),
        (pl.col("log_price_td").shift(-20) - pl.col("log_price_td"))
        .over("ticker_id")
        .alias("target_20d"),
    )

    targets = trading_only.select(
        "ticker_id", "date", "target_1d", "target_5d", "target_20d"
    )
    df = df.join(targets, on=["ticker_id", "date"], how="left")

    # ---------------------------------------------------------------------
    # MARKET / SECTOR CONTEXT (daily, known at t)
    # ---------------------------------------------------------------------
    print("Adding market/sector context features…")
    # Use observed log_return_1d, but only for trading rows at date t
    cs_base = df.filter(~pl.col("is_missing").cast(pl.Boolean)).select(
        "date", "sector_id", "log_return_1d"
    )

    # Market index return = median across tickers that traded on date
    mkt = (
        cs_base.group_by("date")
        .agg(pl.median("log_return_1d").alias("mkt_ret_1d"))
        .sort("date")
    )

    # Sector index return = median within sector/date
    sec = (
        cs_base.group_by(["date", "sector_id"])
        .agg(pl.median("log_return_1d").alias("sector_ret_1d"))
        .sort(["sector_id", "date"])
    )

    # Realized vol (20d) for market
    mkt = mkt.with_columns(
        mkt_vol_20d=pl.col("mkt_ret_1d").rolling_std(window_size=20, min_samples=10)
    )

    # Realized vol (20d) for sector (per sector)
    sec = sec.with_columns(
        sector_vol_20d=pl.col("sector_ret_1d")
        .rolling_std(window_size=20, min_samples=10)
        .over("sector_id")
    )

    # Join context
    df = df.join(mkt, on="date", how="left")
    df = df.join(sec, on=["date", "sector_id"], how="left")

    # --- Region inference from ticker suffix (very light heuristic) ---
    # Map common suffixes to a coarse region label (string literals must use pl.lit)
    df = df.with_columns(
        region=pl.when(
            pl.col("ticker").str.ends_with(".DE")
            | pl.col("ticker").str.ends_with(".F")
            | pl.col("ticker").str.ends_with(".BE")
            | pl.col("ticker").str.ends_with(".DU")
        )
        .then(pl.lit("DE"))
        .when(pl.col("ticker").str.ends_with(".HK"))
        .then(pl.lit("HK"))
        .when(
            pl.col("ticker").str.ends_with(".SZ")
            | pl.col("ticker").str.ends_with(".SS")
        )
        .then(pl.lit("CN"))
        .otherwise(pl.lit("US"))
    )

    # Region index return (median of log_return_1d among trading names in region)
    cs_region = df.filter(~pl.col("is_missing").cast(pl.Boolean)).select(
        "date", "region", "log_return_1d"
    )
    reg = (
        cs_region.group_by(["date", "region"])
        .agg(pl.median("log_return_1d").alias("region_ret_1d"))
        .sort(["region", "date"])
    )
    reg = reg.with_columns(
        pl.col("region_ret_1d")
        .rolling_std(window_size=20, min_samples=10)
        .over("region")
        .alias("region_vol_20d")
    )

    # Join region context
    df = df.join(reg, on=["date", "region"], how="left")

    # --- Clean up any accidental duplicate columns from join ---
    for base in ["region_ret_1d", "region_vol_20d"]:
        right = f"{base}_right"
        if right in df.columns and base in df.columns:
            # prefer the non-suffixed version; drop the _right
            df = df.drop(right)
        elif right in df.columns and base not in df.columns:
            df = df.rename({right: base})

    # Optional: Macro join if available
    macro_path = Path("data/external/macro.parquet")
    if macro_path.exists():
        print(f"Joining macro context from {macro_path}…")
        macro = pl.read_parquet(macro_path)
        # expected columns: date, VIX, VSTOXX, US2Y, US10Y, DE2Y, DE10Y (case-insensitive ok)
        # normalize column names
        macro = macro.rename({c: c.upper() for c in macro.columns})
        want = [
            c
            for c in ["DATE", "VIX", "VSTOXX", "US2Y", "US10Y", "DE2Y", "DE10Y"]
            if c in macro.columns
        ]
        if "DATE" not in want:
            print("Macro parquet present but missing DATE column; skipping macro join.")
        else:
            macro = macro.select(want).rename({"DATE": "date"})
            # cast to Float32 except date
            for c in macro.columns:
                if c != "date":
                    macro = macro.with_columns(pl.col(c).cast(pl.Float32))
            df = df.join(macro, on="date", how="left")
    else:
        print(
            "No macro file at data/external/macro.parquet – skipping VIX/yields (proxies still added)."
        )

    # --- Forward-fill features (not labels) and final null handling ---
    target_cols = ["target_1d", "target_5d", "target_20d"]
    EXCLUDE_FROM_FILLS = set(
        [
            "ticker",
            "date",
            "ticker_id",
            "sector_id",
            "is_quarter_end",
            "is_missing",
            "region",
        ]  # <- added "region"
        + target_cols
    )
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FILLS]

    df = df.with_columns(
        pl.col(feature_cols).forward_fill().over("ticker_id")
    ).with_columns(pl.col(feature_cols).fill_null(0.0))

    # Keep only rows where ALL required columns are valid (labels included)
    df = df.drop_nulls()

    df = df.sort("ticker_id", "date").with_columns(
        time_idx=(pl.col("date").rank("ordinal").over("ticker_id") - 1)
    )

    print(f"Feature creation complete. Final shape: {df.shape}")
    if df.height == 0:
        raise ValueError("All data was dropped after feature engineering.")

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
        final_rows = df_final.height
        retention = (final_rows / initial_rows) * 100 if initial_rows > 0 else 0
        print(f"\nData retention: {final_rows}/{initial_rows} ({retention:.1f}%)")
        total_nulls = df_final.null_count().select(pl.sum_horizontal("*")).item()
        print(f"Total null values in final data: {total_nulls}")
        if total_nulls > 0:
            print("Warning: Null values detected in final DataFrame!")
            print(df_final.null_count())
        print(f"\nSaving final processed data to {PROCESSED_PARQUET_PATH}...")
        df_final.write_parquet(PROCESSED_PARQUET_PATH)
    else:
        print("No data left after processing. Parquet file not saved.")
    print("--- Data Pipeline Complete ---")
