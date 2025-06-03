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
    NaNLabelEncoder,
    EncoderNormalizer,
    MultiNormalizer,
)
from pytorch_forecasting.metrics import QuantileLoss

# Import Lightning Callbacks
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

# Import scikit-learn's StandardScaler if we intend to use it
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler


def get_embedding_dims(timeseries_dataset: TimeSeriesDataSet) -> dict:
    """
    Calculates embedding dimensions for categorical features based on vocabulary size.
    Rule: dim = min(round(N^0.25), 32), where N is vocabulary size.
    """
    embedding_dims = {}
    # timeseries_dataset.categorical_encoders is a dict {col_name: encoder_instance}
    # encoder_instance has a 'cardinality' attribute (for NaNLabelEncoder, etc.)
    # timeseries_dataset.categoricals lists the names of categorical columns
    if not timeseries_dataset.categorical_encoders:
        print(
            "No categorical encoders found in TimeSeriesDataSet. Returning empty embedding_dims."
        )
        return {}

    vocabs = {
        col: timeseries_dataset.categorical_encoders[col].cardinality
        for col in timeseries_dataset.categoricals
        if col
        in timeseries_dataset.categorical_encoders  # Ensure encoder exists for the categorical
    }

    for col, vocab_size in vocabs.items():
        if vocab_size == 0:  # Should not happen with proper data/encoders
            print(f"Warning: Vocab size for {col} is 0. Setting embedding dim to 1.")
            embedding_dims[col] = 1
            continue
        dim = min(round(vocab_size**0.25), 32)
        embedding_dims[col] = int(dim)  # Ensure it's an integer
    print(f"Calculated embedding dimensions: {embedding_dims}")
    return embedding_dims


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

    polars_data_df = pl_df.read_parquet(
        processed_data_path
    )  # Renamed to avoid conflict
    print(f"Loaded processed Polars data. Shape: {polars_data_df.shape}")
    # Ensure all column names are strings, as Pytorch Forecasting expects this
    polars_data_df = polars_data_df.rename(
        {col: str(col) for col in polars_data_df.columns}
    )
    data_pd = polars_data_df.to_pandas()
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
            # Ensure categorical columns are strings for pytorch-forecasting
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
                        transformation=None,  # 'standard' is default
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
        else:
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

    # Define training cutoff for splitting data if not all data is for training
    # For now, assume all data_pd is for training or TimeSeriesDataSet handles split internally.
    # A common way is to define a split point based on time_idx
    # training_cutoff = data_pd[time_idx_str].max() - max_prediction_length * 5 # Example: last 5 prediction periods for validation
    # For simplicity, let TimeSeriesDataSet handle validation split via its val_dataloader()
    # The full dataset is passed to TimeSeriesDataSet.
    # val_dataloader will typically take the last max_prediction_length segment from each time series.

    timeseries_dataset = TimeSeriesDataSet(
        data_pd,  # Using the full dataframe
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
        add_target_scales=True,  # Typically True for GroupNormalizer/EncoderNormalizer
        add_encoder_length=True,
        allow_missing_timesteps=True,  # Important for real-world data
        # If you want to define a specific training dataset (e.g. up to a certain time_idx)
        # you would filter data_pd first and pass that, or use the predict_mode=False and then create
        # validation set separately or rely on TimeSeriesDataSet's split mechanism.
        # For now, assuming TimeSeriesDataSet's default train/val split logic is used.
    )
    print("TimeSeriesDataSet created successfully (using all data).")

    # Get embedding dimensions
    embedding_dims = get_embedding_dims(timeseries_dataset)

    model_module = hydra.utils.get_class(cfg.model._target_)
    model_specific_params_from_cfg = {
        k: v
        for k, v in cfg.model.items()
        if k not in ["_target_", "learning_rate", "weight_decay"]
    }

    # Instantiate loss if configured via _target_
    if "loss" in model_specific_params_from_cfg and isinstance(
        model_specific_params_from_cfg["loss"], DictConfig
    ):
        print("Instantiating loss function from config...")
        model_specific_params_from_cfg["loss"] = hydra.utils.instantiate(
            model_specific_params_from_cfg["loss"]
        )
        print(f"Loss function instantiated: {model_specific_params_from_cfg['loss']}")

    # Add calculated embedding_sizes to model parameters
    model_specific_params_from_cfg["embedding_sizes"] = embedding_dims

    model = model_module(  # Instantiate GlobalTFT
        timeseries_dataset=timeseries_dataset,  # Pass the full dataset for model setup
        model_specific_params=model_specific_params_from_cfg,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )
    print(f"Model {cfg.model._target_} initialized.")

    # Balanced sampling per ticker for training
    # Use the full data_pd to calculate weights, as timeseries_dataset was built on it
    # The sampler will then be used by the training dataloader
    # Note: 'ticker_id' should be one of the group_ids_list and present in data_pd
    ticker_id_col_for_sampler = (
        group_ids_list[0] if group_ids_list else None
    )  # Assuming first group_id is ticker_id
    if not ticker_id_col_for_sampler or ticker_id_col_for_sampler not in data_pd:
        print(
            f"Warning: Ticker ID column '{ticker_id_col_for_sampler}' for sampler not found. Using simple dataloader."
        )
        train_loader = timeseries_dataset.to_dataloader(
            train=True,
            batch_size=cfg.trainer.batch_size,
            num_workers=cfg.trainer.num_workers,
            shuffle=True,
        )
    else:
        print(f"Setting up balanced sampler for column: {ticker_id_col_for_sampler}")
        # Ensure we are referencing the indices that are part of the training set
        # TimeSeriesDataSet.data["data"] holds the pandas dataframe it uses internally.
        # And it has a 'time_idx_first_prediction' attribute.
        # For WeightedRandomSampler, we need weights for each sample in the *training portion* of the dataset.
        # The `TimeSeriesDataSet.to_dataloader(train=True)` internally handles which indices are for training.
        # The challenge is `WeightedRandomSampler` needs weights for the *indices of the dataset object*, not the raw pandas df.

        # Let's get the training indices from the TimeSeriesDataSet
        # This is a bit indirect. Pytorch Forecasting handles this internally.
        # A simpler approach if the TimeSeriesDataSet's internal splitting is opaque for this:
        # Create a training-only TimeSeriesDataSet if strict balancing on *only* training samples is needed.
        # However, the prompt's snippet implies using the full `data_pd` for weights calculation.
        # This might slightly over/under sample boundary cases but is simpler.

        # The provided snippet:
        # weights = data_pd['ticker_id'].value_counts().reindex(data_pd['ticker_id']).rdiv(1.0)
        # sampler = torch.utils.data.WeightedRandomSampler(weights.values, num_samples=len(weights), replacement=True)
        # train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, sampler=sampler, num_workers=4)
        # Here, `train_ds` would be the `timeseries_dataset` if it's only training data, or a subset.
        # And `weights.values` should correspond to the items in `train_ds`.

        # If `timeseries_dataset` is used for both train and val, its indices [0...len(dataset)-1] are used.
        # The `sampler` must provide indices for THIS dataset.
        # So, the weights should correspond to `timeseries_dataset.data["data"][ticker_id_col_for_sampler]`.
        # This internal dataframe is already filtered for `min_encoder_length`, etc.

        internal_df_for_sampler = timeseries_dataset.data[
            "data"
        ]  # This is the DataFrame used by TimeSeriesDataSet

        if ticker_id_col_for_sampler not in internal_df_for_sampler.columns:
            print(
                f"Critical: Ticker ID column '{ticker_id_col_for_sampler}' not in TimeSeriesDataSet's internal DataFrame. Cannot use sampler."
            )
            train_loader = timeseries_dataset.to_dataloader(
                train=True,
                batch_size=cfg.trainer.batch_size,
                num_workers=cfg.trainer.num_workers,
                shuffle=True,
            )
        else:
            counts = internal_df_for_sampler[ticker_id_col_for_sampler].value_counts()
            # Create weights for each row in internal_df_for_sampler
            # Map the counts back to each row based on its ticker_id value
            # 1 / count_for_ticker_of_this_row
            weights_for_rows = internal_df_for_sampler[ticker_id_col_for_sampler].map(
                counts
            )
            weights = 1.0 / weights_for_rows
            # weights.values will be an array of weights, one for each sample in internal_df_for_sampler
            # WeightedRandomSampler expects weights for indices [0, ..., N-1]
            # TimeSeriesDataSet.to_dataloader(train=True) will select a subset of these indices for training.
            # This means the sampler should ideally operate on the *training indices only*.

            # Let's create the train dataloader without the sampler first to get training indices
            # This is a workaround. A cleaner way is to create a separate train TimeSeriesDataSet.
            # For now, let's proceed with the spirit of the prompt, assuming `timeseries_dataset` is effectively the train set for sampler.
            # This means the sampler will sample from the *entire* `timeseries_dataset`, and `to_dataloader(train=True)`
            # will *then* filter those sampled batches for actual training samples. This is inefficient.

            # Correct approach: The sampler should be passed to `to_dataloader`.
            # `to_dataloader` will then use this sampler on the *training indices it determines*.
            # The weights provided to `WeightedRandomSampler` must be for *all* N samples in `timeseries_dataset`.
            # The `num_samples` argument of `WeightedRandomSampler` should be the number of *training samples*.

            # Get train indices directly if possible, or use len(timeseries_dataset) as an approximation for num_samples if sampler re-weights.
            # The `TimeSeriesDataSet` does not easily expose its train/val split indices before `to_dataloader` is called.
            # The simplest interpretation of the prompt is to create weights for *all* samples in `data_pd` (or `timeseries_dataset.data["data"]`)
            # and let the sampler draw `num_samples` from that.

            value_counts = data_pd[ticker_id_col_for_sampler].value_counts()
            sample_weights = (
                data_pd[ticker_id_col_for_sampler]
                .map(value_counts)
                .rdiv(1.0)
                .fillna(1.0)
                .values
            )  # fillna for safety

            # We need to ensure that these weights correspond to the items that `timeseries_dataset` will consider for training.
            # The `timeseries_dataset` object itself has `len(timeseries_dataset)` items.
            # These items are derived from `data_pd` after filtering (e.g. for `min_encoder_length`).
            # The weights for `WeightedRandomSampler` should correspond to these `len(timeseries_dataset)` items.

            # Let's use the `internal_df_for_sampler` which is what TSDataSet uses.
            value_counts_internal = internal_df_for_sampler[
                ticker_id_col_for_sampler
            ].value_counts()
            weights_internal = (
                internal_df_for_sampler[ticker_id_col_for_sampler]
                .map(value_counts_internal)
                .rdiv(1.0)
                .fillna(1.0)
            )

            # The sampler needs weights for each of the dataset's items.
            # If the dataset is 'internal_df_for_sampler', then weights_internal.values is correct.
            # And num_samples should be the number of training samples.
            # `to_dataloader` with `train=True` will select training indices.
            # The sampler samples from the *indices of the dataset object*.

            # The `train_indices` are `self.index[self.index.time_idx_first_prediction <= self.training_cutoff]`
            # Let's assume `num_samples` in `WeightedRandomSampler` should be the size of the training set.
            # And the `weights` array should be for *all* items in `timeseries_dataset`.
            # `to_dataloader` will handle applying the sampler only to the training part.

            # The weights array passed to WeightedRandomSampler must have one entry per sample in the Dataset.
            # `timeseries_dataset` is the dataset.
            # The `weights_internal.values` are derived from `internal_df_for_sampler` which IS the basis of `timeseries_dataset`.
            # So `len(weights_internal.values)` == `len(timeseries_dataset)`. This seems correct.

            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights_internal.values,
                num_samples=len(
                    weights_internal
                ),  # Sample N times, where N is total dataset size. Dataloader will pick train subset.
                # Or, more accurately, num_samples = number of training samples.
                # For now, let len(weights_internal) be used.
                replacement=True,
            )
            print(
                f"WeightedRandomSampler created with {len(weights_internal.values)} weights."
            )

            train_loader = timeseries_dataset.to_dataloader(
                train=True,  # Critical: ensure this selects only training samples
                batch_size=cfg.trainer.batch_size,
                sampler=sampler,  # Pass the sampler here
                num_workers=cfg.trainer.num_workers,
                # shuffle must be False when sampler is provided
            )
            print("Training DataLoader with balanced sampler created.")

    # Validation DataLoader
    val_loader = timeseries_dataset.to_dataloader(
        train=False,  # For validation data
        batch_size=cfg.trainer.batch_size * 2,  # Often use larger batch for validation
        num_workers=cfg.trainer.num_workers,
        shuffle=False,  # No need to shuffle validation data
    )
    print("Validation DataLoader created.")

    # Callbacks
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

    # Logger
    logger = None
    if cfg.trainer.use_wandb:
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            project=cfg.trainer.wandb_project_name,
            entity=cfg.trainer.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print("WandB Logger initialized.")
    else:
        print("WandB Logger is disabled.")

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=(
            cfg.trainer.devices if cfg.trainer.devices != "auto" else "auto"
        ),  # handle "auto" explicitly for devices
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=0.1,  # Common for TFT
        # deterministic=True, # For full reproducibility, but can slow down
    )

    print("Trainer initialized. Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training finished.")


if __name__ == "__main__":
    main()
