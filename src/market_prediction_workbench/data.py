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


# src/market_prediction_workbench/data.py

# src/market_prediction_workbench/data.py


def create_features_and_targets(df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates all model features and targets.
    - Targets: Log-returns for 1, 5, 20 day horizons.
    - Known-future features: Calendar features.
    - Observed-past features: Returns, Volatility, RSI, MACD, etc.
    """
    print("Creating features and targets...")

    df = df.sort("ticker_id", "date")

    log_close = pl.col("close").log()

    # --- Target Definition ---
    df = df.with_columns(
        (log_close.shift(-1) - log_close).over("ticker_id").alias("target_1d"),
        (log_close.shift(-5) - log_close).over("ticker_id").alias("target_5d"),
        (log_close.shift(-20) - log_close).over("ticker_id").alias("target_20d"),
    )

    # --- Feature Engineering ---
    # 1. Known-future (calendar) features
    df = df.with_columns(
        day_of_week=pl.col("date").dt.weekday(),
        day_of_month=pl.col("date").dt.day(),
        month=pl.col("date").dt.month(),
        is_quarter_end=(pl.col("date").dt.month().is_in([3, 6, 9, 12]))
        & (pl.col("date").dt.month_end() == pl.col("date")),
    )

    # 2. Observed-past features
    log_return_1d_base_expr = log_close - log_close.shift(1)

    df = df.with_columns(
        log_return_1d=log_return_1d_base_expr.over("ticker_id"),
        log_return_5d=(log_close - log_close.shift(5)).over("ticker_id"),
        log_return_20d=(log_close - log_close.shift(20)).over("ticker_id"),
        volume_zscore_20d=(
            (
                pl.col("volume")
                - pl.col("volume").rolling_mean(window_size=20).over("ticker_id")
            )
            / (pl.col("volume").rolling_std(window_size=20).over("ticker_id") + 1e-6)
        ).alias("volume_zscore_20d"),
        volatility_20d=log_return_1d_base_expr.rolling_std(window_size=20).over(
            "ticker_id"
        ),
        skew_20d=log_return_1d_base_expr.rolling_skew(window_size=20).over("ticker_id"),
        kurtosis_20d=log_return_1d_base_expr.rolling_kurtosis(window_size=20).over(
            "ticker_id"
        ),
        price_change=pl.col("close").diff(1).over("ticker_id"),
    )

    # RSI calculation
    df = df.with_columns(
        gain=pl.when(pl.col("price_change") > 0)
        .then(pl.col("price_change"))
        .otherwise(0)
        .alias("gain"),
        loss=pl.when(pl.col("price_change") < 0)
        .then(-pl.col("price_change"))
        .otherwise(0)
        .alias("loss"),
    )

    df = df.with_columns(
        avg_gain=pl.col("gain")
        .ewm_mean(alpha=1 / 14, min_samples=14)
        .over("ticker_id"),  # Changed min_periods to min_samples
        avg_loss=pl.col("loss")
        .ewm_mean(alpha=1 / 14, min_samples=14)
        .over("ticker_id"),  # Changed min_periods to min_samples
    )

    rs = pl.col("avg_gain") / (pl.col("avg_loss") + 1e-6)
    df = df.with_columns(rsi_14d=(100 - (100 / (1 + rs))).alias("rsi_14d"))

    # MACD
    ema_12_expr = (
        pl.col("close").ewm_mean(span=12, min_samples=12).over("ticker_id")
    )  # Changed min_periods to min_samples
    ema_26_expr = (
        pl.col("close").ewm_mean(span=26, min_samples=26).over("ticker_id")
    )  # Changed min_periods to min_samples

    # Step 1: Create the 'macd' column
    df = df.with_columns(macd=(ema_12_expr - ema_26_expr).alias("macd"))

    # Step 2: Create the 'macd_signal' column using the now existing 'macd' column
    df = df.with_columns(
        macd_signal=pl.col("macd")
        .ewm_mean(span=9, min_samples=9)
        .over("ticker_id")  # Changed min_periods to min_samples
    )

    # --- Final Cleanup ---
    df = df.drop_nulls().drop(["price_change", "gain", "loss", "avg_gain", "avg_loss"])
    df = df.sort("ticker_id", "date")

    print(f"Feature creation complete. Final shape: {df.shape}")

    df = df.with_columns(
        time_idx=(
            pl.col("date").rank("ordinal").over("ticker_id") - 1
        )  # rank is fine here
    )

    return df


# Update the __main__ block again
if __name__ == "__main__":
    DATA_DIR = Path("data")
    RAW_DATA_PATH = DATA_DIR / "raw" / "stock_data.csv"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    # Define the final processed file path
    PROCESSED_PARQUET_PATH = PROCESSED_DATA_DIR / "processed_data.parquet"

    # This is our full pipeline
    print("--- Starting Data Pipeline ---")
    df = load_and_clean_data(RAW_DATA_PATH)
    df = create_mappings(df, PROCESSED_DATA_DIR)
    df = reindex_and_fill_gaps(df)
    df_final = create_features_and_targets(df)

    print("\n--- Final Processed DataFrame ---")
    print(df_final.head())
    print(f"\nSaving final processed data to {PROCESSED_PARQUET_PATH}...")
    df_final.write_parquet(PROCESSED_PARQUET_PATH)
    print("--- Data Pipeline Complete ---")
