# src/market_prediction_workbench/train.py
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl
import polars as pl_df  # Renamed to avoid conflict with pytorch_lightning.pl
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import shutil
from hydra.core.hydra_config import HydraConfig

# Import our custom modules
from market_prediction_workbench.model import GlobalTFT

# Import pytorch-forecasting specific items
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import (
    GroupNormalizer,
    NaNLabelEncoder,
    EncoderNormalizer,
    MultiNormalizer,
)

# Import Lightning Callbacks
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

# Import scikit-learn's StandardScaler if we intend to use it
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

if torch.cuda.is_available():
    try:
        if torch.cuda.get_device_capability()[0] >= 7:
            torch.set_float32_matmul_precision("medium")
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


# RESTORED: Full, verbose, and robust helper function from your original file
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


def split_before(ds: TimeSeriesDataSet, pct: float = 0.8):
    cutoff = int(ds.data[ds.time_idx].max() * pct)
    train_df = ds.data[ds.data[ds.time_idx] <= cutoff]
    val_df = ds.data[ds.data[ds.time_idx] > cutoff]
    return (
        TimeSeriesDataSet.from_dataset(ds, train_df),
        TimeSeriesDataSet.from_dataset(ds, val_df),
    )


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
    data_pd = polars_data_df.to_pandas()
    print(f"Loaded and converted to Pandas DataFrame. Shape: {data_pd.shape}")

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
                and not pd.api.types.is_string_dtype(data_pd[cat_col_name_str])
            ):
                print(
                    f"Casting categorical column '{cat_col_name_str}' to string. Original dtype: {data_pd[cat_col_name_str].dtype}"
                )
                data_pd[cat_col_name_str] = data_pd[cat_col_name_str].astype(str)
        else:
            print(
                f"Warning: Configured categorical column '{cat_col_name_str}' not found in DataFrame for dtype casting."
            )

    # --- CORRECTED: Split DataFrame BEFORE creating TimeSeriesDataSet objects ---
    max_time_idx = data_pd[time_idx_str].max()
    train_cutoff_idx = int(max_time_idx * 0.8)
    print(f"Splitting data for training/validation at time_idx: {train_cutoff_idx}")

    train_df = data_pd[data_pd[time_idx_str] <= train_cutoff_idx]
    val_df = data_pd[data_pd[time_idx_str] > train_cutoff_idx]

    print(f"Training DataFrame shape: {train_df.shape}")
    print(f"Validation DataFrame shape: {val_df.shape}")

    # Common parameters for both datasets
    dataset_params = dict(
        time_idx=time_idx_str,
        target=target_list[0] if len(target_list) == 1 else target_list,
        group_ids=group_ids_list,
        max_encoder_length=cfg.data.lookback_days,
        max_prediction_length=cfg.data.max_prediction_horizon,
        static_categoricals=static_categoricals_list,
        static_reals=static_reals_list,
        time_varying_known_categoricals=time_varying_known_categoricals_list,
        time_varying_known_reals=time_varying_known_reals_list,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals_list,
        time_varying_unknown_reals=time_varying_unknown_reals_list,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # Scaler and Normalizer Logic (applied to training_dataset first)
    scalers = {}
    if cfg.data.get("scalers") and cfg.data.scalers.get("default_reals_normalizer"):
        default_normalizer_name = cfg.data.scalers.default_reals_normalizer
        reals_to_scale = list(
            dict.fromkeys(
                time_varying_unknown_reals_list
                + time_varying_known_reals_list
                + static_reals_list
            )
        )
        for col in reals_to_scale:
            if default_normalizer_name == "GroupNormalizer":
                scalers[col] = GroupNormalizer(
                    groups=group_ids_list, method="standard"
                )
            elif default_normalizer_name == "EncoderNormalizer":
                scalers[col] = EncoderNormalizer()
            elif default_normalizer_name == "StandardScaler":
                scalers[col] = SklearnStandardScaler()
    else:
        print(
            "No 'default_reals_normalizer' specified. Using GroupNormalizer as default."
        )
        reals_to_scale = list(
            dict.fromkeys(
                time_varying_unknown_reals_list
                + time_varying_known_reals_list
                + static_reals_list
            )
        )
        for col in reals_to_scale:
            scalers[str(col)] = GroupNormalizer(
                groups=group_ids_list, method="standard"
            )

    single_target_normalizer_prototype_name = OmegaConf.select(
        cfg, "data.scalers.target_normalizer", default="GroupNormalizer"
    )
    final_target_normalizer = None
    if single_target_normalizer_prototype_name:
        if len(target_list) > 1:
            normalizers_list = []
            for _ in target_list:
                if single_target_normalizer_prototype_name == "GroupNormalizer":
                    normalizers_list.append(
                        GroupNormalizer(
                            groups=group_ids_list,
                            method="standard",
                        )
                    )
                elif single_target_normalizer_prototype_name == "EncoderNormalizer":
                    normalizers_list.append(EncoderNormalizer())
                elif single_target_normalizer_prototype_name == "StandardScaler":
                    normalizers_list.append(SklearnStandardScaler())
            if normalizers_list:
                final_target_normalizer = MultiNormalizer(normalizers=normalizers_list)
        elif target_list:
            if single_target_normalizer_prototype_name == "GroupNormalizer":
                final_target_normalizer = GroupNormalizer(
                    groups=group_ids_list, method="standard"
                )
            elif single_target_normalizer_prototype_name == "EncoderNormalizer":
                final_target_normalizer = EncoderNormalizer()
            elif single_target_normalizer_prototype_name == "StandardScaler":
                final_target_normalizer = SklearnStandardScaler()

    # Create training dataset. This dataset "learns" the scalers.
    print("Creating training TimeSeriesDataSet...")
    training_dataset = TimeSeriesDataSet(
        train_df,
        **dataset_params,
        scalers=scalers,
        target_normalizer=final_target_normalizer,
    )
    print("Training TimeSeriesDataSet created successfully.")

    # Create validation dataset from the training dataset to ensure scalers are reused.
    print("Creating validation TimeSeriesDataSet from training dataset...")
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, val_df, allow_missing_timesteps=True
    )
    print("Validation TimeSeriesDataSet created successfully.")

    if len(training_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError(
            "Train/validation split resulted in an empty dataset. Check split logic and data range."
        )
    print(
        f"Training samples: {len(training_dataset)}, Validation samples: {len(validation_dataset)}"
    )

    calculated_embedding_sizes = get_embedding_sizes_for_tft(training_dataset)

    model_module = hydra.utils.get_class(cfg.model._target_)
    model_specific_params_from_cfg = {
        k: v
        for k, v in cfg.model.items()
        if k not in ["_target_", "learning_rate", "weight_decay"]
    }

    if "loss" in model_specific_params_from_cfg and isinstance(
        model_specific_params_from_cfg["loss"], DictConfig
    ):
        model_specific_params_from_cfg["loss"] = hydra.utils.instantiate(
            model_specific_params_from_cfg["loss"]
        )

    model_specific_params_from_cfg["embedding_sizes"] = calculated_embedding_sizes

    model = model_module(
        timeseries_dataset=training_dataset,  # Initialize model with training dataset parameters
        model_specific_params=model_specific_params_from_cfg,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )
    print(f"Model {cfg.model._target_} (GlobalTFT wrapper) initialized.")

    # REMOVED WeightedRandomSampler logic
    print("Using default shuffling for train_loader.")
    num_cpu = os.cpu_count()
    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=cfg.trainer.batch_size,
        num_workers=(
            min(num_cpu - 2, cfg.trainer.num_workers)
            if num_cpu and num_cpu > 2
            else cfg.trainer.num_workers
        ),
        shuffle=True,
        pin_memory=True,
        persistent_workers=True if cfg.trainer.num_workers > 0 else False,
        prefetch_factor=4,
    )

    val_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=cfg.trainer.batch_size * 2,
        num_workers=cfg.trainer.num_workers,
        shuffle=False,
        drop_last=False,
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
        dirpath=None,
        filename="{epoch}-{val_loss:.2f}-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    callbacks = [early_stop_callback, lr_monitor, checkpoint_callback]

    logger = None
    if cfg.trainer.get("use_wandb", False):
        from pytorch_lightning.loggers import WandbLogger

        run_name_wandb = f"{cfg.project_name}_{cfg.experiment_id}"
        logger = WandbLogger(
            name=run_name_wandb,
            project=cfg.trainer.wandb_project_name,
            entity=cfg.trainer.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            save_dir=str(Path(cfg.paths.log_dir) / "wandb"),
        )
        print("WandB Logger initialized.")
        if logger.log_dir:
            wandb_run_dir = Path(logger.log_dir)
            hydra_cfg_path = Path(HydraConfig.get().runtime.output_dir) / ".hydra"
            target_hydra_path = wandb_run_dir / ".hydra"
            print(
                f"Copying Hydra config from {hydra_cfg_path} to {target_hydra_path}..."
            )
            try:
                if target_hydra_path.exists():
                    shutil.rmtree(target_hydra_path)
                shutil.copytree(hydra_cfg_path, target_hydra_path)
                print("Successfully copied .hydra config directory.")
            except Exception as e:
                print(f"Error copying .hydra directory: {e}")
        else:
            print("Warning: Could not determine logger.log_dir. Skipping config copy.")
    else:
        print("WandB Logger is disabled.")

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
        num_sanity_val_steps=0,
    )

    print("Trainer initialized. Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training finished.")


if __name__ == "__main__":
    main()
