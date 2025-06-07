# src/market_prediction_workbench/train.py
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl
import polars as pl_df  # Renamed to avoid conflict with pytorch_lightning.pl
import pandas as pd
from pathlib import Path
import numpy as np
import torch  # Added for torch.utils.data.WeightedRandomSampler and DataLoader
from torch.utils.data import DataLoader  # Added for DataLoader
import os
import shutil  # For copying files/directories
from hydra.core.hydra_config import HydraConfig  # To get Hydra's output path

# Import our custom modules
from market_prediction_workbench.model import GlobalTFT

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
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

# Import scikit-learn's StandardScaler if we intend to use it
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler


torch.backends.cudnn.benchmark = True  # autotune conv/LSTM kernels
torch.backends.cuda.matmul.allow_tf32 = True  # use TF32 tensor cores


# Set matmul precision for Tensor Cores if applicable
# This was the fix for the previous RuntimeError regarding device mismatch during matmul/attention
if torch.cuda.is_available():
    # Check if the GPU supports Tensor Cores (Compute Capability 7.0+)
    # This is a heuristic; specific precision ('high' or 'medium') might depend on the exact GPU and desired trade-off.
    # For RTX 4090 (Ada Lovelace), CC is 8.9, so this will apply.
    try:
        if torch.cuda.get_device_capability()[0] >= 7:
            torch.set_float32_matmul_precision("medium")  # or 'high'
            print("PyTorch float32 matmul precision set to 'medium' for Tensor Cores.")
        else:
            print(
                "Current GPU does not have Tensor Cores (or CC < 7.0). Default matmul precision used."
            )
    except Exception as e:
        print(
            f"Could not set matmul precision (may be normal if no CUDA GPU or older PyTorch): {e}"
        )
else:
    print("CUDA not available. Running on CPU. Matmul precision setting skipped.")


# MODIFIED function to be more robust
def get_embedding_sizes_for_tft(timeseries_dataset: TimeSeriesDataSet) -> dict:
    embedding_sizes = {}

    # Access the internal attribute _categorical_encoders, which is populated by TimeSeriesDataSet
    dataset_encoders = timeseries_dataset._categorical_encoders

    # Check if _categorical_encoders attribute exists and is a non-empty dictionary
    if not isinstance(dataset_encoders, dict) or not dataset_encoders:
        if (
            timeseries_dataset.categoricals
        ):  # Check if there are any categoricals defined in the dataset
            print(
                "Warning (get_embedding_sizes_for_tft): TimeSeriesDataSet._categorical_encoders is missing, not a dict, or empty, "
                "but dataset has categorical columns. TFT might use defaults or error."
            )
        return {}  # Return empty if no encoders or not a dict

    print(
        f"DEBUG (get_embedding_sizes_for_tft): Processing encoders from TimeSeriesDataSet._categorical_encoders: {dataset_encoders}"
    )

    for col_name in (
        timeseries_dataset.categoricals
    ):  # Iterate over actual categoricals defined in dataset
        if col_name in dataset_encoders:
            encoder = dataset_encoders[col_name]
            print(
                f"DEBUG (get_embedding_sizes_for_tft): Encoder for '{col_name}': {encoder}, type: {type(encoder)}"
            )

            cardinality_val = None
            # Try .cardinality property first (NaNLabelEncoder has this)
            if hasattr(encoder, "cardinality"):
                try:
                    cardinality_val = encoder.cardinality
                    if cardinality_val is not None:
                        print(
                            f"DEBUG (get_embedding_sizes_for_tft): Accessed encoder.cardinality for '{col_name}': {cardinality_val}"
                        )
                    else:
                        print(
                            f"DEBUG (get_embedding_sizes_for_tft): encoder.cardinality for '{col_name}' returned None."
                        )
                except AttributeError:
                    print(
                        f"DEBUG (get_embedding_sizes_for_tft): AttributeError on encoder.cardinality for '{col_name}'. Will try .classes_."
                    )
                    cardinality_val = None

            if cardinality_val is None:  # Fallback or if .cardinality was None
                if hasattr(encoder, "classes_") and encoder.classes_ is not None:
                    num_classes = len(encoder.classes_)
                    add_nan_flag = False
                    # Check for add_nan attribute (specific to NaNLabelEncoder but good general check)
                    # NaNLabelEncoder is the main one that uses add_nan and contributes to cardinality this way
                    if hasattr(encoder, "add_nan") and isinstance(
                        encoder, NaNLabelEncoder
                    ):
                        add_nan_flag = encoder.add_nan

                    cardinality_val = num_classes + (1 if add_nan_flag else 0)

                    print(
                        f"DEBUG (get_embedding_sizes_for_tft): Calculated cardinality from len(encoder.classes_) for '{col_name}': {cardinality_val} (add_nan={add_nan_flag})"
                    )
                else:
                    print(
                        f"ERROR (get_embedding_sizes_for_tft): Could not determine cardinality for '{col_name}'. Skipping."
                    )
                    continue

            if cardinality_val is None:  # Should not happen if logic above is complete
                print(
                    f"ERROR (get_embedding_sizes_for_tft): Cardinality for '{col_name}' is unexpectedly None. Skipping."
                )
                continue

            # For TFT, cardinality must be at least 1.
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
                f"not found in TimeSeriesDataSet._categorical_encoders. This is unexpected if encoders were meant to be created for all."
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
            # Check if not already object, string, or pandas dedicated string type
            if (
                data_pd[cat_col_name_str].dtype != object
                and data_pd[cat_col_name_str].dtype
                != str  # Python's built-in str type for columns
                and not pd.api.types.is_string_dtype(
                    data_pd[cat_col_name_str]
                )  # Pandas' extension string dtype
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
            "StandardScaler",  # This refers to our SklearnStandardScaler wrapper
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
        reals_to_scale = list(dict.fromkeys(reals_to_scale))  # Unique columns

        for col_name_str_loop_var in reals_to_scale:
            current_col_name = str(col_name_str_loop_var)
            if current_col_name in data_pd.columns:
                if default_normalizer_name == "GroupNormalizer":
                    scalers[current_col_name] = GroupNormalizer(
                        groups=group_ids_list,  # Ensure group_ids_list is correctly populated
                        transformation=None,  # Or specify transformation if needed
                    )
                elif default_normalizer_name == "EncoderNormalizer":
                    scalers[current_col_name] = EncoderNormalizer()
                elif default_normalizer_name == "StandardScaler":
                    scalers[current_col_name] = SklearnStandardScaler()
    else:
        print(
            "No 'default_reals_normalizer' specified in cfg.data.scalers. PTF will use its defaults for feature scaling."
        )

    single_target_normalizer_prototype_name = "GroupNormalizer"  # Default
    if (
        OmegaConf.select(cfg, "data.scalers.target_normalizer") is not None
    ):  # Check existence properly
        single_target_normalizer_prototype_name = cfg.data.scalers.target_normalizer

    valid_target_normalizer_names = [
        "GroupNormalizer",
        "EncoderNormalizer",
        "StandardScaler",  # This refers to scikit-learn's StandardScaler
        "NaNLabelEncoder",  # If target can be categorical
        None,  # To explicitly use no normalizer (PTF will use TorchNormalizer(method="identity"))
    ]
    if (
        single_target_normalizer_prototype_name not in valid_target_normalizer_names
        and single_target_normalizer_prototype_name is not None
    ):
        print(
            f"Warning: Unknown target_normalizer '{single_target_normalizer_prototype_name}'. Using GroupNormalizer as default."
        )
        single_target_normalizer_prototype_name = "GroupNormalizer"

    final_target_normalizer = None
    if single_target_normalizer_prototype_name is None:
        final_target_normalizer = None  # PTF default
    elif len(target_list) > 1:
        list_of_normalizers_for_multi = []
        for _ in target_list:  # Create a distinct normalizer instance for each target
            if single_target_normalizer_prototype_name == "GroupNormalizer":
                list_of_normalizers_for_multi.append(
                    GroupNormalizer(groups=group_ids_list, transformation=None)
                )
            elif single_target_normalizer_prototype_name == "EncoderNormalizer":
                list_of_normalizers_for_multi.append(EncoderNormalizer())
            elif single_target_normalizer_prototype_name == "StandardScaler":
                list_of_normalizers_for_multi.append(SklearnStandardScaler())
            elif single_target_normalizer_prototype_name == "NaNLabelEncoder":
                list_of_normalizers_for_multi.append(NaNLabelEncoder())
        if list_of_normalizers_for_multi:
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
        elif single_target_normalizer_prototype_name == "NaNLabelEncoder":
            final_target_normalizer = NaNLabelEncoder()

    if (
        not final_target_normalizer
        and target_list
        and single_target_normalizer_prototype_name is not None
    ):
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
        categorical_encoders=None,
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
    print(
        f"  DEBUG: TimeSeriesDataSet._categorical_encoders (before calling get_embedding_sizes_for_tft): {timeseries_dataset._categorical_encoders}"
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
        decoded_idx_df = timeseries_dataset.decoded_index
        if ticker_id_col_for_sampler in decoded_idx_df.columns:
            value_counts_sampler = decoded_idx_df[
                ticker_id_col_for_sampler
            ].value_counts()

            if not value_counts_sampler.empty:
                weights_map_sampler = 1.0 / value_counts_sampler
                weights_for_rows_sampler = (
                    decoded_idx_df[ticker_id_col_for_sampler]
                    .map(weights_map_sampler)
                    .fillna(1.0)
                )

                if (weights_for_rows_sampler.values <= 0).any():
                    print(
                        "Warning: Some calculated weights for sampler are non-positive. Clamping to a small positive value."
                    )
                    weights_for_rows_sampler.values[
                        weights_for_rows_sampler.values <= 0
                    ] = 1e-6

                if len(weights_for_rows_sampler) == len(timeseries_dataset):
                    sampler = torch.utils.data.WeightedRandomSampler(
                        weights=weights_for_rows_sampler.values,
                        num_samples=len(weights_for_rows_sampler),
                        replacement=True,
                    )

                    num_cpu = os.cpu_count()
                    train_loader = timeseries_dataset.to_dataloader(
                        train=True,
                        batch_size=cfg.trainer.batch_size,
                        sampler=sampler,
                        num_workers=(
                            min(num_cpu - 2, cfg.trainer.num_workers)
                            if num_cpu and num_cpu > 2
                            else cfg.trainer.num_workers
                        ),
                        shuffle=False,
                        pin_memory=True,
                        persistent_workers=(
                            True if cfg.trainer.num_workers > 0 else False
                        ),
                        prefetch_factor=4,  # default is 2 – double it
                    )
                    train_loader_created_with_sampler = True
                    print(
                        f"Train Dataloader created with WeightedRandomSampler for '{ticker_id_col_for_sampler}'."
                    )
                else:
                    print(
                        f"Warning: Length of weights ({len(weights_for_rows_sampler)}) does not match dataset length ({len(timeseries_dataset)}). "
                        "Sampler not used."
                    )
            else:
                print(
                    f"Warning: Value counts for '{ticker_id_col_for_sampler}' in decoded_index_df is empty. Sampler not used."
                )
        else:
            print(
                f"Critical: Ticker ID column '{ticker_id_col_for_sampler}' not in TimeSeriesDataSet's decoded_index. Sampler not used."
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

    checkpoint_callback = ModelCheckpoint(
        dirpath=None,  # If None, PTL saves to its default logger path (e.g., lightning_logs/version_X/checkpoints)
        # You can specify a path like "my_checkpoints/"
        filename="{epoch}-{val_loss:.2f}-best",  # Example filename
        monitor="val_loss",  # Metric to monitor
        mode="min",  # "min" for loss, "max" for accuracy
        save_top_k=1,  # Save only the best model
        save_last=True,  # Also save the last model at the end of training
        verbose=True,
    )

    callbacks = [early_stop_callback, lr_monitor, checkpoint_callback]

    logger = None
    if cfg.trainer.get("use_wandb", False):
        from pytorch_lightning.loggers import WandbLogger

        project_name_wandb = str(cfg.get("project_name", "default_project"))
        exp_id_wandb = str(cfg.get("experiment_id", "default_exp"))
        run_name_wandb = f"{project_name_wandb}_{exp_id_wandb}"

        logger = WandbLogger(
            name=run_name_wandb,
            project=cfg.trainer.wandb_project_name,
            entity=cfg.trainer.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            save_dir=str(Path(cfg.paths.log_dir) / "wandb"),
        )
        print("WandB Logger initialized.")

        # --- REVISED: COPY HYDRA CONFIG TO WANDB RUN DIRECTORY ---
        # Use logger.log_dir which is the specific directory for this run.
        # This is the same directory ModelCheckpoint uses.
        if logger.log_dir:
            wandb_run_dir = Path(logger.log_dir)
            hydra_cfg_path = Path(HydraConfig.get().runtime.output_dir) / ".hydra"
            target_hydra_path = wandb_run_dir / ".hydra"

            print(
                f"Copying Hydra config from {hydra_cfg_path} to {target_hydra_path}..."
            )
            try:
                if target_hydra_path.exists():
                    print(
                        f".hydra directory already exists at {target_hydra_path}. Overwriting."
                    )
                    shutil.rmtree(target_hydra_path)
                shutil.copytree(hydra_cfg_path, target_hydra_path)
                print("Successfully copied .hydra config directory.")
            except Exception as e:
                print(f"Error copying .hydra directory: {e}")
        else:
            print("Warning: Could not determine logger.log_dir. Skipping config copy.")
        # --- END OF REVISED LOGIC ---

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
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 0.1),
        num_sanity_val_steps=0,  # ← skip the initial sanity‐check validation
    )

    print("Trainer initialized. Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training finished.")


if __name__ == "__main__":
    main()
