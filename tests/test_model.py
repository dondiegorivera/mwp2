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
