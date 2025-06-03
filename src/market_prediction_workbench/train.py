# src/market_prediction_workbench/train.py
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl
import polars as pl_df  # Renamed to avoid conflict with pytorch_lightning.pl
from pathlib import Path
import numpy as np
import torch  # Added for torch.utils.data.WeightedRandomSampler and DataLoader
from torch.utils.data import DataLoader  # Added for DataLoader

# Import our custom modules

# Import pytorch-forecasting specific items
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import (
    GroupNormalizer,
    NaNLabelEncoder,  # Keep this import for type checking if needed
    EncoderNormalizer,
    MultiNormalizer,
)

# from pytorch_forecasting.metrics import QuantileLoss # Instantiated by hydra

# Import Lightning Callbacks
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

# Import scikit-learn's StandardScaler if we intend to use it
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler


# MODIFIED function to be more robust
def get_embedding_sizes_for_tft(timeseries_dataset: TimeSeriesDataSet) -> dict:
    embedding_sizes = {}

    # Check if categorical_encoders attribute exists and is a dictionary (not None)
    if not hasattr(timeseries_dataset, "categorical_encoders") or not isinstance(
        timeseries_dataset.categorical_encoders, dict
    ):
        if (
            timeseries_dataset.categoricals
        ):  # Check if there are any categoricals defined in the dataset
            print(
                "Warning (get_embedding_sizes_for_tft): TimeSeriesDataSet.categorical_encoders is missing, not a dict, or empty, "
                "but dataset has categorical columns. TFT might use defaults or error."
            )
        return {}  # Return empty if no encoders or not a dict

    print(
        f"DEBUG (get_embedding_sizes_for_tft): Processing encoders from TimeSeriesDataSet: {timeseries_dataset.categorical_encoders}"
    )

    # If timeseries_dataset.categorical_encoders is an empty dict, but timeseries_dataset.categoricals is not,
    # it means TSD did not populate the encoders, which is an issue upstream.
    if not timeseries_dataset.categorical_encoders and timeseries_dataset.categoricals:
        print(
            "CRITICAL (get_embedding_sizes_for_tft): TimeSeriesDataSet.categorical_encoders is an empty dictionary, "
            "but timeseries_dataset.categoricals is not. This implies encoders were not created/fitted by TimeSeriesDataSet. "
            "TFT will likely fail."
        )
        return {}

    for col_name in (
        timeseries_dataset.categoricals
    ):  # Iterate over actual categoricals defined in dataset
        if col_name in timeseries_dataset.categorical_encoders:
            encoder = timeseries_dataset.categorical_encoders[col_name]
            print(
                f"DEBUG (get_embedding_sizes_for_tft): Encoder for '{col_name}': {encoder}, type: {type(encoder)}"
            )

            cardinality_val = None
            # Try .cardinality property first
            if hasattr(encoder, "cardinality"):
                try:
                    cardinality_val = encoder.cardinality
                    if cardinality_val is not None:
                        print(
                            f"DEBUG (get_embedding_sizes_for_tft): Accessed encoder.cardinality for '{col_name}': {cardinality_val}"
                        )
                    else:
                        # This means encoder might not be fitted if cardinality property returned None
                        print(
                            f"DEBUG (get_embedding_sizes_for_tft): encoder.cardinality for '{col_name}' returned None."
                        )
                except AttributeError:
                    print(
                        f"DEBUG (get_embedding_sizes_for_tft): AttributeError on encoder.cardinality for '{col_name}'. Will try .classes_."
                    )
                    cardinality_val = None  # Ensure fallback

            # Fallback to .classes_ if .cardinality didn't work or returned None
            if cardinality_val is None:
                if hasattr(encoder, "classes_") and encoder.classes_ is not None:
                    num_classes = len(encoder.classes_)
                    add_nan_flag = False
                    # Check for add_nan attribute (specific to NaNLabelEncoder but good general check)
                    if hasattr(encoder, "add_nan"):
                        add_nan_flag = encoder.add_nan

                    cardinality_val = num_classes + (1 if add_nan_flag else 0)
                    print(
                        f"DEBUG (get_embedding_sizes_for_tft): Calculated cardinality from len(encoder.classes_) for '{col_name}': {cardinality_val}"
                    )
                else:
                    print(
                        f"ERROR (get_embedding_sizes_for_tft): Could not determine cardinality for '{col_name}' from .classes_ either. Skipping."
                    )
                    continue

            # If, after all attempts, cardinality_val is still None (shouldn't happen if logic above is complete)
            if cardinality_val is None:
                print(
                    f"ERROR (get_embedding_sizes_for_tft): Cardinality for '{col_name}' is unexpectedly None. Skipping."
                )
                continue

            # For TFT, cardinality must be at least 1.
            # If len(classes_)=0 and add_nan=True, card will be 1. If add_nan=False, card will be 0.
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
                f"not found in TimeSeriesDataSet.categorical_encoders. This is unexpected if encoders were meant to be created for all."
            )

    if not embedding_sizes and timeseries_dataset.categoricals:
        print(
            "Warning (get_embedding_sizes_for_tft): Resulting embedding_sizes dictionary is empty, but dataset has categoricals. TFT will use defaults or error."
        )
    elif embedding_sizes:
        print(f"Calculated embedding_sizes for TFT: {embedding_sizes}")
    return embedding_sizes


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
    print(f"Loaded processed Polars data. Shape: {polars_data_df.shape}")
    polars_data_df = polars_data_df.rename(
        {col: str(col) for col in polars_data_df.columns}
    )
    data_pd = polars_data_df.to_pandas()
    print(f"Converted to Pandas DataFrame. Shape: {data_pd.shape}")

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
            ):
                print(
                    f"Casting categorical column '{cat_col_name_str}' to string. Original dtype: {data_pd[cat_col_name_str].dtype}"
                )
                data_pd[cat_col_name_str] = data_pd[cat_col_name_str].astype(str)
        else:
            print(
                f"Warning: Configured categorical column '{cat_col_name_str}' not found in DataFrame for dtype casting."
            )

    max_encoder_length = cfg.data.lookback_days
    max_prediction_length = cfg.data.max_prediction_horizon
    scalers = {}
    if cfg.data.get("scalers") and cfg.data.scalers.get("default_reals_normalizer"):
        default_normalizer_name = cfg.data.scalers.default_reals_normalizer
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
        reals_to_scale = list(dict.fromkeys(reals_to_scale))

        for col_name_str_loop_var in reals_to_scale:
            current_col_name = str(col_name_str_loop_var)
            if current_col_name in data_pd.columns:
                if default_normalizer_name == "GroupNormalizer":
                    scalers[current_col_name] = GroupNormalizer(
                        groups=group_ids_list,
                        transformation=None,
                    )
                elif default_normalizer_name == "EncoderNormalizer":
                    scalers[current_col_name] = EncoderNormalizer()
                elif default_normalizer_name == "StandardScaler":
                    scalers[current_col_name] = SklearnStandardScaler()
    else:
        print(
            "No 'default_reals_normalizer' specified. PTF will use its defaults for feature scaling."
        )

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

    final_target_normalizer = None
    if len(target_list) > 1:
        list_of_normalizers_for_multi = []
        for _ in target_list:
            if single_target_normalizer_prototype_name == "GroupNormalizer":
                list_of_normalizers_for_multi.append(
                    GroupNormalizer(groups=group_ids_list, transformation=None)
                )
            elif single_target_normalizer_prototype_name == "EncoderNormalizer":
                list_of_normalizers_for_multi.append(EncoderNormalizer())
            elif single_target_normalizer_prototype_name == "StandardScaler":
                list_of_normalizers_for_multi.append(SklearnStandardScaler())
        if list_of_normalizers_for_multi:
            final_target_normalizer = MultiNormalizer(
                normalizers=list_of_normalizers_for_multi
            )
    elif target_list:
        if single_target_normalizer_prototype_name == "GroupNormalizer":
            final_target_normalizer = GroupNormalizer(
                groups=group_ids_list, transformation=None
            )
        elif single_target_normalizer_prototype_name == "EncoderNormalizer":
            final_target_normalizer = EncoderNormalizer()
        elif single_target_normalizer_prototype_name == "StandardScaler":
            final_target_normalizer = SklearnStandardScaler()

    if not final_target_normalizer and target_list:
        print(
            f"Warning: Target normalizer could not be constructed for {single_target_normalizer_prototype_name} and targets {target_list}. Check logic."
        )
    elif not target_list:
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

    print(
        f"Instantiating TimeSeriesDataSet. Static categoricals list from config: {static_categoricals_list}"
    )
    timeseries_dataset = TimeSeriesDataSet(
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
        categorical_encoders=None,  # <<<<<<<< MODIFICATION: Leave as None so PTF auto-creates encoders
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    print("TimeSeriesDataSet created successfully (using all data).")
    print(
        f"  DEBUG: TimeSeriesDataSet.static_categoricals (property): {timeseries_dataset.static_categoricals}"
    )
    print(
        f"  DEBUG: TimeSeriesDataSet.categoricals (property): {timeseries_dataset.categoricals}"
    )

    # Force encoder creation/fitting by calling get_embedding_sizes()
    _ = timeseries_dataset.get_embedding_sizes()
    print(
        f"  DEBUG: TimeSeriesDataSet.categorical_encoders (property) after get_embedding_sizes(): {timeseries_dataset.categorical_encoders}"
    )

    calculated_embedding_sizes = get_embedding_sizes_for_tft(timeseries_dataset)

    model_module = hydra.utils.get_class(cfg.model._target_)
    model_specific_params_from_cfg = {
        k: v
        for k, v in cfg.model.items()
        if k not in ["_target_", "learning_rate", "weight_decay"]
    }

    if "loss" in model_specific_params_from_cfg and isinstance(
        model_specific_params_from_cfg["loss"], DictConfig
    ):
        print("Instantiating loss function from config...")
        model_specific_params_from_cfg["loss"] = hydra.utils.instantiate(
            model_specific_params_from_cfg["loss"]
        )
        print(f"Loss function instantiated: {model_specific_params_from_cfg['loss']}")

    model_specific_params_from_cfg["embedding_sizes"] = calculated_embedding_sizes

    model = model_module(
        timeseries_dataset=timeseries_dataset,
        model_specific_params=model_specific_params_from_cfg,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )
    print(f"Model {cfg.model._target_} (GlobalTFT wrapper) initialized.")

    ticker_id_col_for_sampler = group_ids_list[0] if group_ids_list else None
    train_loader_created_with_sampler = False

    if ticker_id_col_for_sampler and ticker_id_col_for_sampler in data_pd:
        internal_df_for_sampler = timeseries_dataset.data["data"]

        if ticker_id_col_for_sampler in internal_df_for_sampler.columns:
            value_counts_internal = internal_df_for_sampler[
                ticker_id_col_for_sampler
            ].value_counts()

            if not value_counts_internal.empty:
                weights_internal_map = value_counts_internal.rdiv(1.0)
                weights_for_rows = (
                    internal_df_for_sampler[ticker_id_col_for_sampler]
                    .map(weights_internal_map)
                    .fillna(1.0)
                )

                if (weights_for_rows.values <= 0).any():
                    print(
                        "Warning: Some calculated weights for sampler are non-positive. Clamping to a small positive value."
                    )
                    weights_for_rows.values[weights_for_rows.values <= 0] = 1e-6

                if len(weights_for_rows.values) == len(timeseries_dataset):
                    num_samples_for_sampler = len(weights_for_rows)

                    sampler = torch.utils.data.WeightedRandomSampler(
                        weights=weights_for_rows.values,
                        num_samples=num_samples_for_sampler,
                        replacement=True,
                    )
                    train_loader = timeseries_dataset.to_dataloader(
                        train=True,
                        batch_size=cfg.trainer.batch_size,
                        sampler=sampler,
                        num_workers=cfg.trainer.num_workers,
                    )
                    train_loader_created_with_sampler = True
                else:
                    print(
                        f"Warning: Length of weights ({len(weights_for_rows.values)}) does not match dataset length ({len(timeseries_dataset)}). "
                        "Sampler not used."
                    )
            else:
                print(
                    f"Warning: Value counts for '{ticker_id_col_for_sampler}' in internal_df_for_sampler is empty. Sampler not used."
                )
        else:
            print(
                f"Critical: Ticker ID column '{ticker_id_col_for_sampler}' not in TimeSeriesDataSet's internal DataFrame. Sampler not used."
            )

    if not train_loader_created_with_sampler:
        print(
            f"Warning: Balanced sampler for column '{ticker_id_col_for_sampler}' could not be set up. Using default shuffling for train_loader."
        )
        train_loader = timeseries_dataset.to_dataloader(
            train=True,
            batch_size=cfg.trainer.batch_size,
            num_workers=cfg.trainer.num_workers,
            shuffle=True,
        )

    val_loader = timeseries_dataset.to_dataloader(
        train=False,
        batch_size=cfg.trainer.batch_size * 2,
        num_workers=cfg.trainer.num_workers,
        shuffle=False,
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
    callbacks = [early_stop_callback, lr_monitor]

    logger = None
    if cfg.trainer.get("use_wandb", False):
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            project=cfg.trainer.wandb_project_name,
            entity=cfg.trainer.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
        )
        print("WandB Logger initialized.")
    else:
        print("WandB Logger is disabled (use_wandb=false or key missing).")

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
        gradient_clip_val=0.1,
    )

    print("Trainer initialized. Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training finished.")


if __name__ == "__main__":
    main()
