# src/market_prediction_workbench/train.py
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl
import polars as pl_df  # Renamed to avoid conflict with pytorch_lightning.pl
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import shutil
from hydra.core.hydra_config import HydraConfig

from market_prediction_workbench.model import GlobalTFT

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import (
    GroupNormalizer,
    NaNLabelEncoder,
    EncoderNormalizer,
    MultiNormalizer,
)

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

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


def get_embedding_sizes_for_tft(timeseries_dataset: TimeSeriesDataSet) -> dict:
    embedding_sizes = {}
    dataset_encoders = timeseries_dataset._categorical_encoders

    if not isinstance(dataset_encoders, dict) or not dataset_encoders:
        if timeseries_dataset.categoricals:
            print(
                "Warning: _categorical_encoders missing/empty but dataset has categoricals."
            )
        return {}

    for col_name in timeseries_dataset.categoricals:
        if col_name in dataset_encoders:
            encoder = dataset_encoders[col_name]

            cardinality_val = None
            if hasattr(encoder, "cardinality"):
                try:
                    cardinality_val = encoder.cardinality
                except AttributeError:
                    cardinality_val = None

            if cardinality_val is None:
                if hasattr(encoder, "classes_") and encoder.classes_ is not None:
                    num_classes = len(encoder.classes_)
                    add_nan_flag = (
                        hasattr(encoder, "add_nan")
                        and isinstance(encoder, NaNLabelEncoder)
                        and encoder.add_nan
                    )
                    cardinality_val = num_classes + (1 if add_nan_flag else 0)
                else:
                    print(
                        f"ERROR: Could not determine cardinality for '{col_name}'. Skipping."
                    )
                    continue

            tft_cardinality = max(1, cardinality_val)
            if tft_cardinality <= 1:
                dim = 1
            else:
                dim = min(round(tft_cardinality**0.25), 32)
                dim = max(1, int(dim))
            embedding_sizes[col_name] = (tft_cardinality, dim)
            print(f"DEBUG embedding for '{col_name}': ({tft_cardinality}, {dim})")
        else:
            print(
                f"Warning: categorical '{col_name}' missing in _categorical_encoders."
            )

    if not embedding_sizes and timeseries_dataset.categoricals:
        print("Warning: embedding_sizes empty; TFT will use defaults or error.")
    else:
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


def _cfg_list(val):
    if val is None:
        return []
    if isinstance(val, (str, int, float)):
        return [str(val)]
    if isinstance(val, (list, ListConfig)):
        return [str(v) for v in val]
    raise TypeError(f"Unsupported cfg node type: {type(val)}")


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

    # Ensure proper dtypes
    if "date" in data_pd.columns:
        data_pd["date"] = pd.to_datetime(data_pd["date"])
    else:
        raise ValueError("Required column 'date' not found in processed data.")

    time_idx_col_name = (
        str(cfg.data.time_idx) if OmegaConf.select(cfg, "data.time_idx") else "time_idx"
    )
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
    time_idx_str = (
        str(cfg.data.time_idx) if OmegaConf.select(cfg, "data.time_idx") else "time_idx"
    )

    # Cast configured categoricals to str
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

    # --- DATE-BASED SPLIT WITH EMBARGO ---
    if OmegaConf.select(cfg, "data.split.use_date_split", default=True):
        min_date, max_date = data_pd["date"].min(), data_pd["date"].max()
        train_frac = float(OmegaConf.select(cfg, "data.split.train_pct", default=0.8))
        embargo_days = int(OmegaConf.select(cfg, "data.split.embargo_days", default=30))
        cutoff_date = min_date + (max_date - min_date) * train_frac
        embargo = pd.Timedelta(days=embargo_days)

        train_df = data_pd[data_pd["date"] <= (cutoff_date - embargo)]
        val_df = data_pd[data_pd["date"] >= (cutoff_date + embargo)]
        print(
            f"Date split: train ≤ {cutoff_date - embargo:%Y-%m-%d}, val ≥ {cutoff_date + embargo:%Y-%m-%d}"
        )
    else:
        # fallback to time_idx split
        max_time_idx = data_pd[time_idx_str].max()
        train_cutoff_idx = int(max_time_idx * 0.8)
        print(f"Splitting data for training/validation at time_idx: {train_cutoff_idx}")
        train_df = data_pd[data_pd[time_idx_str] <= train_cutoff_idx]
        val_df = data_pd[data_pd[time_idx_str] > train_cutoff_idx]

    print(f"Training DataFrame shape: {train_df.shape}")
    print(f"Validation DataFrame shape: {val_df.shape}")

    # --- ensure validation only contains groups seen in training ---
    # PF encoders/normalizers are fit on training; unseen groups in val will error.
    if "ticker_id" in train_df.columns and "ticker_id" in val_df.columns:
        # both were cast to string above for categoricals – keep consistent
        train_tickers = set(train_df["ticker_id"].astype(str).unique())
        before_rows = len(val_df)
        before_tickers = val_df["ticker_id"].nunique()
        val_df = val_df[val_df["ticker_id"].astype(str).isin(train_tickers)].copy()
        after_rows = len(val_df)
        after_tickers = val_df["ticker_id"].nunique()
        print(
            f"Validation filter: kept {after_rows}/{before_rows} rows; "
            f"tickers {after_tickers}/{before_tickers} overlap with training."
        )

    # --- DATASET PARAMS ---
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

    # --- SCALERS / NORMALIZERS ---
    # exclude booleans and raw calendar integers from per-ticker scaling
    EXCLUDE_FROM_SCALING = {
        "is_quarter_end",
        "is_missing",
        "day_of_week",
        "day_of_month",
        "month",
    }
    scalers = {}
    if cfg.data.get("scalers") and cfg.data.scalers.get("default_reals_normalizer"):
        default_normalizer_name = cfg.data.scalers.default_reals_normalizer
        reals_to_scale_all = list(
            dict.fromkeys(
                time_varying_unknown_reals_list
                + time_varying_known_reals_list
                + static_reals_list
            )
        )
        reals_to_scale = [
            c for c in reals_to_scale_all if c not in EXCLUDE_FROM_SCALING
        ]
        for col in reals_to_scale:
            if default_normalizer_name == "GroupNormalizer":
                scalers[col] = GroupNormalizer(groups=group_ids_list, method="standard")
            elif default_normalizer_name == "EncoderNormalizer":
                scalers[col] = EncoderNormalizer()
            elif default_normalizer_name == "StandardScaler":
                scalers[col] = SklearnStandardScaler()
    else:
        print(
            "No 'default_reals_normalizer' specified. Using GroupNormalizer as default (excluding flags/raw calendars)."
        )
        reals_to_scale_all = list(
            dict.fromkeys(
                time_varying_unknown_reals_list
                + time_varying_known_reals_list
                + static_reals_list
            )
        )
        reals_to_scale = [
            c for c in reals_to_scale_all if c not in EXCLUDE_FROM_SCALING
        ]
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
                        GroupNormalizer(groups=group_ids_list, method="standard")
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

    print("Creating training TimeSeriesDataSet...")
    training_dataset = TimeSeriesDataSet(
        train_df,
        **dataset_params,
        scalers=scalers,
        target_normalizer=final_target_normalizer,
    )
    print("Training TimeSeriesDataSet created successfully.")

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
        timeseries_dataset=training_dataset,
        model_specific_params=model_specific_params_from_cfg,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )
    print(f"Model {cfg.model._target_} (GlobalTFT wrapper) initialized.")

    # --- TRAIN DATALOADER (optional balanced sampling) ---
    num_cpu = os.cpu_count()
    train_loader_kwargs = dict(
        train=True,
        batch_size=cfg.trainer.batch_size,
        num_workers=(
            min(num_cpu - 2, cfg.trainer.num_workers)
            if num_cpu and num_cpu > 2
            else cfg.trainer.num_workers
        ),
        pin_memory=True,
        persistent_workers=True if cfg.trainer.num_workers > 0 else False,
        prefetch_factor=4,
    )

    use_weighted = bool(
        OmegaConf.select(cfg, "trainer.use_weighted_sampler", default=False)
    )
    train_loader = None

    # --- TRAIN DATALOADER (optional balanced sampling) ---
    num_cpu = os.cpu_count()
    train_loader_kwargs = dict(
        train=True,
        batch_size=cfg.trainer.batch_size,
        num_workers=(
            min(num_cpu - 2, cfg.trainer.num_workers)
            if num_cpu and num_cpu > 2
            else cfg.trainer.num_workers
        ),
        pin_memory=True,
        persistent_workers=True if cfg.trainer.num_workers > 0 else False,
        prefetch_factor=4,
    )

    use_weighted = bool(
        OmegaConf.select(cfg, "trainer.use_weighted_sampler", default=False)
    )
    train_loader = None

    if use_weighted:
        try:
            # Fast, vectorized per-sample weights from the dataset's sample index
            idx_df = training_dataset.index
            if idx_df is None:
                raise RuntimeError("training_dataset.index is not available")

            group_col = (
                f"__group_id__{group_ids_list[0]}"  # e.g., "__group_id__ticker_id"
            )
            grp_vals = idx_df[group_col].astype(str)
            counts = idx_df[group_col].value_counts()
            mapped_counts = counts.reindex(grp_vals).to_numpy()
            weights = (
                idx_df[group_col].map(counts).rpow(-1.0).astype(np.float64).values
            )  # 1 / count
            weights_np = (1.0 / mapped_counts).astype("float32")

            sampler = WeightedRandomSampler(
                weights=weights, num_samples=len(weights), replacement=True
            )

            train_loader = training_dataset.to_dataloader(
                train=True,
                batch_size=cfg.trainer.batch_size,
                sampler=sampler,  # <--- use sampler
                shuffle=False,  # <--- must be False when sampler is set
                num_workers=...,
                pin_memory=True,
                persistent_workers=True if cfg.trainer.num_workers > 0 else False,
                prefetch_factor=4,
            )

            print(
                f"Using WeightedRandomSampler over {len(counts)} groups for {len(weights_np)} samples."
            )

        except Exception as e:
            print(f"Weighted sampler setup failed ({e}); falling back to shuffle=True.")
            train_loader = training_dataset.to_dataloader(
                shuffle=True,
                **train_loader_kwargs,
            )
    else:
        print("Using default shuffling for train_loader.")
        train_loader = training_dataset.to_dataloader(
            shuffle=True,
            **train_loader_kwargs,
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
        # Log split stats
        try:
            import wandb

            wandb.run.summary["train_rows"] = len(train_df)
            wandb.run.summary["val_rows"] = len(val_df)
            wandb.run.summary["train_min_date"] = str(train_df["date"].min())
            wandb.run.summary["train_max_date"] = str(train_df["date"].max())
            wandb.run.summary["val_min_date"] = str(val_df["date"].min())
            wandb.run.summary["val_max_date"] = str(val_df["date"].max())
            wandb.run.summary["group_ids"] = group_ids_list
            wandb.run.summary["known_reals"] = time_varying_known_reals_list
            wandb.run.summary["unknown_reals"] = time_varying_unknown_reals_list
        except Exception as e:
            print(f"W&B split logging failed: {e}")
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
