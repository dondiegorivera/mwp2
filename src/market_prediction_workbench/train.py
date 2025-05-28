# src/market_prediction_workbench/train.py
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl
import polars as pl_df
from pathlib import Path

# Import our custom modules

# Import pytorch-forecasting specific items
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import (
    GroupNormalizer,
    NaNLabelEncoder,
    EncoderNormalizer,
)
from pytorch_forecasting.metrics import QuantileLoss

# Import scikit-learn's StandardScaler if we intend to use it
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler


def get_embedding_dims(processed_data_path: Path, data_conf: DictConfig) -> dict:
    # Placeholder - this function is not fully implemented yet
    return {}


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    pl.seed_everything(cfg.seed, workers=True)

    processed_data_path = Path(cfg.paths.processed_data_file)
    if not processed_data_path.exists():
        print(
            f"Processed data not found at {processed_data_path}. Please run data pipeline or enable it in train script."
        )
        return

    data = pl_df.read_parquet(processed_data_path)
    print(f"Loaded processed data. Shape: {data.shape}")

    max_encoder_length = cfg.data.lookback_days
    max_prediction_length = cfg.data.max_prediction_horizon

    # --- Scaler Configuration ---
    scalers = {}
    # Ensure cfg.data.scalers exists and has default_reals_normalizer
    if cfg.data.get("scalers") and cfg.data.scalers.get("default_reals_normalizer"):
        default_normalizer_name = cfg.data.scalers.default_reals_normalizer

        chosen_scaler_instance = None
        if default_normalizer_name == "GroupNormalizer":
            chosen_scaler_instance = GroupNormalizer(
                groups=cfg.data.group_ids, transformation="identity"
            )
        elif default_normalizer_name == "EncoderNormalizer":
            chosen_scaler_instance = EncoderNormalizer()
        elif default_normalizer_name == "StandardScaler":
            chosen_scaler_instance = SklearnStandardScaler()
        else:
            print(
                f"Warning: Unknown default_reals_normalizer '{default_normalizer_name}'. Using EncoderNormalizer."
            )
            chosen_scaler_instance = EncoderNormalizer()

        # Assign the chosen scaler to all relevant real-valued features
        reals_to_scale = (
            (cfg.data.get("time_varying_unknown_reals", []) or [])
            + (cfg.data.get("time_varying_known_reals", []) or [])
            + (cfg.data.get("static_reals", []) or [])
        )
        reals_to_scale = list(dict.fromkeys(reals_to_scale))  # Ensure uniqueness

        for col_name in reals_to_scale:
            if col_name in data.columns:
                # Allow for specific scalers later if defined in cfg.data.scalers.feature_name
                scalers[col_name] = chosen_scaler_instance
            else:
                print(
                    f"Warning: Column '{col_name}' for scaling not found in data. Skipping scaler."
                )

    # --- Target Normalizer Configuration ---
    target_normalizer_instance = GroupNormalizer(
        groups=cfg.data.group_ids, transformation="identity"
    )  # Default
    if cfg.data.get("target_normalizer"):
        target_normalizer_name = cfg.data.target_normalizer
        if target_normalizer_name == "GroupNormalizer":
            target_normalizer_instance = GroupNormalizer(
                groups=cfg.data.group_ids, transformation="identity"
            )
        elif target_normalizer_name == "EncoderNormalizer":
            target_normalizer_instance = EncoderNormalizer()
        elif target_normalizer_name == "StandardScaler":
            target_normalizer_instance = SklearnStandardScaler()
        else:
            print(
                f"Warning: Unknown target_normalizer '{target_normalizer_name}'. Using GroupNormalizer."
            )

    print("Creating TimeSeriesDataSet...")

    # Consolidate all feature column names from config, ensuring they are lists
    def get_list_from_cfg(config_node, key_name):
        val = config_node.get(key_name, [])  # Default to empty list if key missing
        if val is None:
            return []
        if isinstance(val, (str, int, float)):
            return [val]  # Single item treat as list
        if isinstance(val, (list, ListConfig)):
            return list(val)
        raise TypeError(f"Expected list or primitive for {key_name}, got {type(val)}")

    group_ids_list = get_list_from_cfg(cfg.data, "group_ids")
    target_list = get_list_from_cfg(cfg.data, "target")
    static_categoricals_list = get_list_from_cfg(cfg.data, "static_categoricals")
    static_reals_list = get_list_from_cfg(cfg.data, "static_reals")
    time_varying_known_categoricals_list = get_list_from_cfg(
        cfg.data, "time_varying_known_categoricals"
    )
    time_varying_known_reals_list = get_list_from_cfg(
        cfg.data, "time_varying_known_reals"
    )
    time_varying_unknown_categoricals_list = get_list_from_cfg(
        cfg.data, "time_varying_unknown_categoricals"
    )
    time_varying_unknown_reals_list = get_list_from_cfg(
        cfg.data, "time_varying_unknown_reals"
    )

    all_config_cols = set(
        group_ids_list
        + [cfg.data.time_idx]  # time_idx is a string, not a list
        + target_list
        + static_categoricals_list
        + static_reals_list
        + time_varying_known_categoricals_list
        + time_varying_known_reals_list
        + time_varying_unknown_categoricals_list
        + time_varying_unknown_reals_list
    )

    for col_name in all_config_cols:
        if col_name not in data.columns:
            raise ValueError(
                f"Configuration error: Column '{col_name}' (from cfg.data) not found in DataFrame. Available: {data.columns}"
            )

    categorical_encoders = {}
    cat_cols_to_check = (
        static_categoricals_list
        + time_varying_known_categoricals_list
        + time_varying_unknown_categoricals_list
    )
    cat_cols_to_check = list(dict.fromkeys(cat_cols_to_check))

    for cat_col in cat_cols_to_check:
        if cat_col in data.columns and data[cat_col].null_count() > 0:
            if (
                data[cat_col].dtype.is_integer()
                or data[cat_col].dtype.is_unsigned_integer()
                or data[cat_col].dtype == pl_df.Utf8
                or data[cat_col].dtype == pl_df.Categorical
            ):
                categorical_encoders[cat_col] = NaNLabelEncoder(add_nan=True)
            else:
                print(
                    f"Warning: Column {cat_col} has NaNs but unsuitable dtype for NaNLabelEncoder. Skipping."
                )

    # For splitting data for train/validation:
    # This is a placeholder. A robust split is needed.
    # E.g., by date:
    # unique_dates = sorted(data[cfg.data.time_idx].unique().to_list())
    # if not unique_dates: raise ValueError("No time_idx values found for splitting data")
    # training_cutoff_idx_val = unique_dates[int(len(unique_dates) * 0.8)] # 80% train

    # Or, for TFT, often we want a validation set that's a continuation of encoder data
    # training_cutoff_time_idx = data[cfg.data.time_idx].max() - max_prediction_length * 5 # e.g. last 5 prediction lengths for validation
    # This is better than a random split or simple fraction for time series.
    # Let's use this simple cutoff for now for testing purposes.
    # You'll need to ensure 'time_idx' is numeric for this. If it's date, convert or use date logic.
    # Assuming cfg.data.time_idx refers to our numeric 'time_idx' column.

    # For initial run, we'll create one dataset from all data to check instantiation
    # Later, split data into train_df, val_df then create respective TimeSeriesDataSets
    print(
        f"Using max_encoder_length: {max_encoder_length}, max_prediction_length: {max_prediction_length}"
    )

    temp_dataset_for_params = TimeSeriesDataSet(
        data,  # Using entire dataset for now for param extraction
        time_idx=cfg.data.time_idx,
        target=target_list,
        group_ids=group_ids_list,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals_list,
        static_reals=static_reals_list,
        time_varying_known_categoricals=time_varying_known_categoricals_list,
        time_varying_known_reals=time_varying_known_reals_list,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals_list,
        time_varying_unknown_reals=time_varying_unknown_reals_list,
        target_normalizer=target_normalizer_instance,
        scalers=scalers if scalers else None,
        categorical_encoders=categorical_encoders if categorical_encoders else None,
        add_relative_time_idx=True,
        add_target_scales=(
            True
            if isinstance(
                target_normalizer_instance, (GroupNormalizer, EncoderNormalizer)
            )
            else False
        ),
        add_encoder_length=True,
        allow_missing_timesteps=True,  # We ffilled, but some series might start late.
    )

    print(
        "TimeSeriesDataSet created successfully (using all data for param extraction)."
    )

    # --- Model Initialization ---
    model_module = hydra.utils.get_class(cfg.model._target_)

    # Prepare model_specific_params from cfg.model, excluding Hydra/Lightning specific ones
    model_specific_params_from_cfg = {
        k: v
        for k, v in cfg.model.items()
        if k not in ["_target_", "learning_rate", "weight_decay"]
    }

    # Add quantiles/loss from data config to model_specific_params for TFT
    if hasattr(cfg.data, "target_quantiles") and cfg.data.target_quantiles:
        model_specific_params_from_cfg["loss"] = QuantileLoss(
            quantiles=cfg.data.target_quantiles
        )
        # output_size is usually num_quantiles * num_targets.
        # If target is a list, from_dataset might infer num_targets.
        # If target is a single string but represents multiple actual values (not typical for PTF),
        # output_size needs care. Assuming target_list contains the names of output variables.
        model_specific_params_from_cfg["output_size"] = len(
            cfg.data.target_quantiles
        ) * len(target_list)
        # For multiple targets, PTF TFT concatenates their quantile predictions.
        # If each target in target_list is a single variable, then output_size should be just len(cfg.data.target_quantiles)
        # and TFT's target parameter should be a list of these columns.
        # The `target_multi` argument in TFT `from_dataset` might be relevant.
        # Let's assume for now `target_list` names the actual target columns, and TFT handles multiple targets.
        # If `target_list` has 3 targets, and `quantiles` has 3 values, `output_size` might be 3 for TFT
        # (meaning 3 quantiles per target), and the model's output tensor would be [batch, time, 3*num_targets].
        # Let's simplify: output_size = number of quantiles. TFT will handle replicating this for multiple targets.
        model_specific_params_from_cfg["output_size"] = len(cfg.data.target_quantiles)

    # Example for embedding_dims (if we want to override TimeSeriesDataSet's defaults)
    # embedding_dims_calculated = get_embedding_dims(processed_data_path, cfg.data)
    # if embedding_dims_calculated:
    #    model_specific_params_from_cfg["embedding_sizes"] = embedding_dims_calculated
    # PTF's from_dataset uses "embedding_sizes" not "embedding_dims"

    model = model_module(
        timeseries_dataset=temp_dataset_for_params,  # Pass the dataset
        model_specific_params=model_specific_params_from_cfg,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )

    print(f"Model {cfg.model._target_} initialized.")

    # --- Trainer and Fit (Placeholder) ---
    # print("Initializing PyTorch Lightning Trainer...")
    # trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True)) # Convert trainer config to dict
    # print("Trainer initialized.")
    # print("Starting training...")
    # Placeholder for creating actual train/val dataloaders from split data
    # train_loader = ...
    # val_loader = ...
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # print("Training complete.")


if __name__ == "__main__":
    main()
