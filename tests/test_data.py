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
