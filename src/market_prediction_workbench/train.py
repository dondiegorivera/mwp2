# src/market_prediction_workbench/train.py
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl
import polars as pl_df
from pathlib import Path
import numpy as np  # Added for np.int64

# Import our custom modules

# Import pytorch-forecasting specific items
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import (
    GroupNormalizer,
    NaNLabelEncoder,
    EncoderNormalizer,
    MultiNormalizer,
)
from pytorch_forecasting.metrics import QuantileLoss

# Import scikit-learn's StandardScaler if we intend to use it
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler


def get_embedding_dims(processed_data_path: Path, data_conf: DictConfig) -> dict:
    # Placeholder
    return {}


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------")

    pl.seed_everything(cfg.seed, workers=True)

    processed_data_path = Path(cfg.paths.processed_data_file)
    if not processed_data_path.exists():
        print(f"Processed data not found at {processed_data_path}.")
        return

    polars_data = pl_df.read_parquet(processed_data_path)
    print(f"Loaded processed Polars data. Shape: {polars_data.shape}")
    polars_data = polars_data.rename({col: str(col) for col in polars_data.columns})
    data_pd = polars_data.to_pandas()
    print(f"Converted to Pandas DataFrame. Shape: {data_pd.shape}")

    # Cast time_idx to int64
    time_idx_col_name = str(cfg.data.time_idx)
    if time_idx_col_name in data_pd.columns:
        print(
            f"Original dtype of '{time_idx_col_name}': {data_pd[time_idx_col_name].dtype}"
        )
        data_pd[time_idx_col_name] = data_pd[time_idx_col_name].astype(np.int64)
        print(
            f"Casted dtype of '{time_idx_col_name}': {data_pd[time_idx_col_name].dtype}"
        )
    else:
        raise ValueError(
            f"Time index column '{time_idx_col_name}' not found for casting."
        )

    # Helper to get lists from config, ensuring they are actual Python lists of strings
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

    for cat_col_name_str in static_categoricals_list:
        if cat_col_name_str in data_pd.columns:
            print(
                f"Original dtype of static_categorical '{cat_col_name_str}': {data_pd[cat_col_name_str].dtype}"
            )
            data_pd[cat_col_name_str] = data_pd[cat_col_name_str].astype(str)
            print(
                f"Casted dtype of static_categorical '{cat_col_name_str}': {data_pd[cat_col_name_str].dtype}"
            )
        else:
            print(
                f"Warning: Static categorical column '{cat_col_name_str}' not found for dtype casting."
            )

    max_encoder_length = cfg.data.lookback_days
    max_prediction_length = cfg.data.max_prediction_horizon

    scalers = {}
    if cfg.data.get("scalers") and cfg.data.scalers.get("default_reals_normalizer"):
        default_normalizer_name = cfg.data.scalers.default_reals_normalizer

        # Validate normalizer name, and set a fallback if necessary
        valid_normalizer_names = [
            "GroupNormalizer",
            "EncoderNormalizer",
            "StandardScaler",
        ]
        if default_normalizer_name not in valid_normalizer_names:
            print(
                f"Warning: Unknown default_reals_normalizer '{default_normalizer_name}'. Defaulting to EncoderNormalizer."
            )
            default_normalizer_name = "EncoderNormalizer"

        reals_to_scale = (
            time_varying_unknown_reals_list
            + time_varying_known_reals_list
            + static_reals_list
        )
        reals_to_scale = list(
            dict.fromkeys(reals_to_scale)
        )  # Remove duplicates, preserve order

        for (
            col_name_str_loop_var
        ) in reals_to_scale:  # Renamed loop variable for clarity
            # Ensure col_name_str_loop_var is a string, though it should be from get_list_from_cfg_node
            current_col_name = str(col_name_str_loop_var)
            if current_col_name in data_pd.columns:
                if default_normalizer_name == "GroupNormalizer":
                    # transformation=None is from original code (via MODIFIED comment)
                    # This implies GroupNormalizer uses its default 'standard' method.
                    scalers[current_col_name] = GroupNormalizer(
                        groups=group_ids_list, transformation=None
                    )
                elif default_normalizer_name == "EncoderNormalizer":
                    scalers[current_col_name] = EncoderNormalizer()
                elif default_normalizer_name == "StandardScaler":
                    scalers[current_col_name] = SklearnStandardScaler()
                # No 'else' needed here because default_normalizer_name is guaranteed to be one of the valid names.
    else:
        print(
            "No 'default_reals_normalizer' specified. PTF will use its defaults for feature scaling."
        )

    # Initialize with a sensible default before checking config
    # Default to GroupNormalizer based on original code logic/comments
    single_target_normalizer_prototype_name = "GroupNormalizer"
    if cfg.data.get("target_normalizer"):
        single_target_normalizer_prototype_name = cfg.data.target_normalizer

    valid_target_normalizer_names = [
        "GroupNormalizer",
        "EncoderNormalizer",
        "StandardScaler",
    ]
    if single_target_normalizer_prototype_name not in valid_target_normalizer_names:
        print(
            f"Warning: Unknown target_normalizer '{single_target_normalizer_prototype_name}'. Using GroupNormalizer as default."
        )
        single_target_normalizer_prototype_name = "GroupNormalizer"

    if len(target_list) > 1:
        list_of_normalizers_for_multi = []
        for _ in target_list:  # Create a new instance for each target
            if single_target_normalizer_prototype_name == "GroupNormalizer":
                list_of_normalizers_for_multi.append(
                    GroupNormalizer(groups=group_ids_list, transformation=None)
                )
            elif single_target_normalizer_prototype_name == "EncoderNormalizer":
                list_of_normalizers_for_multi.append(EncoderNormalizer())
            elif single_target_normalizer_prototype_name == "StandardScaler":
                list_of_normalizers_for_multi.append(SklearnStandardScaler())
        final_target_normalizer = MultiNormalizer(
            normalizers=list_of_normalizers_for_multi
        )
    elif target_list:  # Single target
        if single_target_normalizer_prototype_name == "GroupNormalizer":
            final_target_normalizer = GroupNormalizer(
                groups=group_ids_list, transformation=None
            )
        elif single_target_normalizer_prototype_name == "EncoderNormalizer":
            final_target_normalizer = EncoderNormalizer()
        elif single_target_normalizer_prototype_name == "StandardScaler":
            final_target_normalizer = SklearnStandardScaler()
        else:  # Should not be reached due to earlier validation
            final_target_normalizer = GroupNormalizer(
                groups=group_ids_list, transformation=None
            )
    else:
        final_target_normalizer = None
        print(
            "Warning: No targets defined in cfg.data.target. Target normalizer set to None."
        )

    print("Creating TimeSeriesDataSet...")

    all_config_cols_set = set(
        group_ids_list
        + [time_idx_str]
        + target_list
        + static_categoricals_list
        + static_reals_list
        + time_varying_known_categoricals_list
        + time_varying_known_reals_list
        + time_varying_unknown_categoricals_list
        + time_varying_unknown_reals_list
    )

    for col_name_check in all_config_cols_set:
        if col_name_check not in data_pd.columns:
            raise ValueError(
                f"Configuration error: Column '{col_name_check}' from config not in DataFrame. Available: {list(data_pd.columns)}"
            )

    categorical_encoders = {}
    cat_cols_to_check = (
        static_categoricals_list
        + time_varying_known_categoricals_list
        + time_varying_unknown_categoricals_list
    )
    cat_cols_to_check = list(dict.fromkeys(cat_cols_to_check))

    for cat_col in cat_cols_to_check:
        if cat_col in data_pd.columns and data_pd[cat_col].isnull().any():
            categorical_encoders[str(cat_col)] = NaNLabelEncoder(add_nan=True)

    print(
        f"Using max_encoder_length: {max_encoder_length}, max_prediction_length: {max_prediction_length}"
    )
    print(f"Scalers to be used: {scalers}")
    print(f"Categorical encoders: {categorical_encoders}")
    print(f"Final Target normalizer: {final_target_normalizer}")

    # Determine add_target_scales based on the type of the final_target_normalizer
    # or its components if it's a MultiNormalizer
    add_target_scales_flag = False
    if final_target_normalizer is not None:
        if isinstance(final_target_normalizer, (GroupNormalizer, EncoderNormalizer)):
            add_target_scales_flag = True
        elif isinstance(final_target_normalizer, MultiNormalizer) and all(
            isinstance(n, (GroupNormalizer, EncoderNormalizer))
            for n in final_target_normalizer.normalizers
        ):
            add_target_scales_flag = True

    temp_dataset_for_params = TimeSeriesDataSet(
        data_pd,
        time_idx=time_idx_str,
        target=target_list[0] if len(target_list) == 1 else target_list,
        group_ids=group_ids_list,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals_list,
        static_reals=static_reals_list,
        time_varying_known_categoricals=time_varying_known_categoricals_list,
        time_varying_known_reals=time_varying_known_reals_list,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals_list,
        time_varying_unknown_reals=time_varying_unknown_reals_list,
        target_normalizer=final_target_normalizer,
        scalers=scalers if scalers else {},
        categorical_encoders=categorical_encoders if categorical_encoders else {},
        add_relative_time_idx=True,
        add_target_scales=add_target_scales_flag,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    print(
        "TimeSeriesDataSet created successfully (using all data for param extraction)."
    )

    model_module = hydra.utils.get_class(cfg.model._target_)
    model_specific_params_from_cfg = {
        k: v
        for k, v in cfg.model.items()
        if k not in ["_target_", "learning_rate", "weight_decay"]
    }

    if hasattr(cfg.data, "target_quantiles") and cfg.data.target_quantiles:
        quantiles_list_cfg = get_list_from_cfg_node(cfg.data.target_quantiles)
        quantiles_float_list = [float(q) for q in quantiles_list_cfg]
        model_specific_params_from_cfg["loss"] = QuantileLoss(
            quantiles=quantiles_float_list
        )

    model = model_module(  # Instantiate GlobalTFT directly
        timeseries_dataset=temp_dataset_for_params,
        model_specific_params=model_specific_params_from_cfg,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )

    print(f"Model {cfg.model._target_} initialized.")


if __name__ == "__main__":
    main()
