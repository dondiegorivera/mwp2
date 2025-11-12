# src/market_prediction_workbench/train.py
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl
import polars as pl_df  # Renamed to avoid conflict with pytorch_lightning.pl
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
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
from pytorch_forecasting.data.encoders import (
    GroupNormalizer as _PFGroupNorm,
    MultiNormalizer as _PFMultiNorm,
)

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
import math


@torch.no_grad()
def _inverse_with_groups(
    data: torch.Tensor, normalizer, groups: torch.Tensor
) -> torch.Tensor:
    """
    Invert PF target normalizer per group for correct cross-sectional ranking.
    """
    if isinstance(normalizer, _PFGroupNorm):
        g = groups[:, 0].cpu().numpy()
        scale = torch.as_tensor(
            normalizer.get_parameters(g), dtype=data.dtype, device=data.device
        )
        loc, sigm = scale[:, 0], scale[:, 1]
        while loc.dim() < data.dim():
            loc = loc.unsqueeze(1)
        while sigm.dim() < data.dim():
            sigm = sigm.unsqueeze(1)
        return data * sigm + loc
    if isinstance(normalizer, _PFMultiNorm):
        parts = [
            _inverse_with_groups(data[..., i], sub, groups)
            for i, sub in enumerate(normalizer.normalizers)
        ]
        return torch.stack(parts, dim=-1)
    # Fallback: plain inverse (rare for targets)
    out = normalizer.inverse_transform(data)
    return torch.as_tensor(out, dtype=data.dtype, device=data.device)


def _cfg_list(val):
    if val is None:
        return []
    if isinstance(val, (str, int, float)):
        return [str(val)]
    if isinstance(val, (list, ListConfig)):
        return [str(v) for v in val]
    raise TypeError(f"Unsupported cfg node type: {type(val)}")


class RankICCallback(pl.callbacks.Callback):
    """
    Computes daily Spearman rank-IC on validation data (horizon=1, target_idx=0)
    after each val epoch and logs `val_rank_ic`.
    Uses the dataset's target_normalizer to invert predictions per-group safely.
    Optionally restricts to the last N decoder days (calendar index) to stabilize IC.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataSet,
        target_idx: int = 0,
        horizon: int = 1,
        last_n_days: int | None = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.target_idx = int(target_idx)
        self.horizon = int(horizon)
        self.last_n_days = (
            int(last_n_days) if last_n_days and int(last_n_days) > 0 else None
        )

    @staticmethod
    def _inverse_with_groups(
        data: torch.Tensor, normalizer, groups: torch.Tensor
    ) -> torch.Tensor:
        from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer

        if isinstance(normalizer, GroupNormalizer):
            g = groups[:, 0].detach().cpu().numpy()
            scale = torch.as_tensor(
                normalizer.get_parameters(g), dtype=data.dtype, device=data.device
            )
            loc, sig = scale[:, 0], scale[:, 1]
            while loc.dim() < data.dim():
                loc = loc.unsqueeze(1)
            while sig.dim() < data.dim():
                sig = sig.unsqueeze(1)
            return data * sig + loc
        if isinstance(normalizer, MultiNormalizer):
            parts = [
                RankICCallback._inverse_with_groups(data[..., i], sub, groups)
                for i, sub in enumerate(normalizer.normalizers)
            ]
            return torch.stack(parts, dim=-1)
        out = normalizer.inverse_transform(data)
        return torch.as_tensor(out, dtype=data.dtype, device=data.device)

    @staticmethod
    def _spearman(x: np.ndarray, y: np.ndarray) -> float:
        xr = pd.Series(x).rank(method="average").to_numpy()
        yr = pd.Series(y).rank(method="average").to_numpy()
        xv = xr - xr.mean()
        yv = yr - yr.mean()
        denom = np.sqrt((xv**2).sum()) * np.sqrt((yv**2).sum())
        if denom <= 0 or len(xv) < 2:
            return np.nan
        return float((xv * yv).sum() / denom)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module.eval()
        device = pl_module.device
        loader = (
            trainer.val_dataloaders[0]
            if isinstance(trainer.val_dataloaders, (list, tuple))
            else trainer.val_dataloaders
        )
        tft = getattr(pl_module, "model", pl_module)

        # median quantile index
        q_list = getattr(getattr(tft, "loss", None), "quantiles", None)
        mid_idx = 0
        if isinstance(q_list, (list, tuple)):
            try:
                q_vals = np.array([float(q) for q in q_list], dtype=float)
                mid_idx = int(np.argmin(np.abs(q_vals - 0.5)))
            except Exception:
                pass

        preds_all, trues_all, days_all = [], [], []

        with torch.no_grad():
            for batch in loader:
                x, y = batch
                # move dict of tensors
                for k, v in x.items():
                    if torch.is_tensor(v):
                        x[k] = v.to(device)

                y_norm = y[0] if isinstance(y, (list, tuple)) else y
                if y_norm.dim() == 2:
                    y_norm = y_norm.unsqueeze(2)  # [B,H,1] -> [B,H,T]

                out = tft(x)
                pred_norm = out.prediction
                if isinstance(pred_norm, list):
                    pred_norm = torch.stack(pred_norm, dim=2).unsqueeze(-1)  # [B,H,T,1]
                elif pred_norm.dim() == 3:
                    pred_norm = pred_norm.unsqueeze(2)  # [B,H,1,Q]

                pred_sel = pred_norm[:, self.horizon - 1, self.target_idx, mid_idx]
                true_sel = y_norm[:, self.horizon - 1, self.target_idx]

                groups = x["groups"]
                normalizer = self.dataset.target_normalizer
                pred_dec = self._inverse_with_groups(pred_sel, normalizer, groups)
                true_dec = self._inverse_with_groups(true_sel, normalizer, groups)

                dti = x.get("decoder_time_idx", None)
                if dti is None:
                    dti = x.get("time", None)
                if dti is None:
                    continue
                day = dti[:, self.horizon - 1].detach().cpu().numpy()

                preds_all.append(pred_dec.detach().cpu().numpy())
                trues_all.append(true_dec.detach().cpu().numpy())
                days_all.append(day)

        if not preds_all:
            pl_module.log("val_rank_ic", float("nan"), prog_bar=True, on_epoch=True)
            return

        preds = np.concatenate(preds_all, axis=0)
        trues = np.concatenate(trues_all, axis=0)
        days = np.concatenate(days_all, axis=0)

        df = pd.DataFrame({"day": days, "p": preds, "y": trues})

        # keep only last N days if requested
        if self.last_n_days:
            max_day = int(np.nanmax(df["day"].values))
            min_keep = max_day - self.last_n_days + 1
            df = df[df["day"] >= min_keep]

        ics = []
        for _, sub in df.groupby("day"):
            p = sub["p"].to_numpy()
            y = sub["y"].to_numpy()
            m = np.isfinite(p) & np.isfinite(y)
            if m.sum() >= 5:
                ics.append(self._spearman(p[m], y[m]))

        ic_mean = float(np.nanmean(ics)) if len(ics) > 0 else float("nan")
        pl_module.log("val_rank_ic", ic_mean, prog_bar=True, on_epoch=True)


# --------------------------------------------------------------------- #
# Torch perf knobs                                                      #
# --------------------------------------------------------------------- #
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

            tft_cardinality = max(1, int(cardinality_val))
            if tft_cardinality <= 1:
                dim = 1
            else:
                dim = int(min(64, math.ceil(math.sqrt(tft_cardinality))))

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


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #
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
        cutoff_override = OmegaConf.select(cfg, "data.split.cutoff_date")
        val_end_override = OmegaConf.select(cfg, "data.split.val_end_date")
        if cutoff_override:
            cutoff_date = pd.to_datetime(str(cutoff_override))
        else:
            cutoff_date = min_date + (max_date - min_date) * train_frac
        val_end_date = (
            pd.to_datetime(str(val_end_override)) if val_end_override else max_date
        )
        embargo = pd.Timedelta(days=embargo_days)

        train_df = data_pd[data_pd["date"] <= (cutoff_date - embargo)]
        val_df = data_pd[
            (data_pd["date"] >= (cutoff_date + embargo))
            & (data_pd["date"] <= val_end_date)
        ]
        print(
            f"Date split: train ≤ {cutoff_date - embargo:%Y-%m-%d}, val ∈ [{cutoff_date + embargo:%Y-%m-%d}, {val_end_date:%Y-%m-%d}]"
        )
    else:
        max_time_idx = data_pd[time_idx_str].max()
        train_cutoff_idx = int(max_time_idx * 0.8)
        print(f"Splitting data for training/validation at time_idx: {train_cutoff_idx}")
        train_df = data_pd[data_pd[time_idx_str] <= train_cutoff_idx]
        val_df = data_pd[data_pd[time_idx_str] > train_cutoff_idx]

    print(f"Training DataFrame shape: {train_df.shape}")
    print(f"Validation DataFrame shape: {val_df.shape}")

    # --- ensure validation only contains groups seen in training ---
    if "ticker_id" in train_df.columns and "ticker_id" in val_df.columns:
        train_tickers = set(train_df["ticker_id"].astype(str).unique())
        before_rows = len(val_df)
        before_tickers = val_df["ticker_id"].nunique()
        val_df = val_df[val_df["ticker_id"].astype(str).isin(train_tickers)].copy()
        after_rows = len(val_df)
        after_tickers = val_df["ticker_id"].nunique()
        print(
            f"Validation filter: kept {after_rows}/{before_rows} rows; tickers {after_tickers}/{before_tickers} overlap with training."
        )

    # ensure we own these frames (avoid pandas view warnings later)
    train_df = train_df.copy()
    val_df = val_df.copy()

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

    # diagnostics
    missing_known = [
        c for c in time_varying_known_reals_list if c not in train_df.columns
    ]
    if missing_known:
        print(f"[warn] time_varying_known_reals not in dataframe: {missing_known}")
    missing_unknown = [
        c for c in time_varying_unknown_reals_list if c not in train_df.columns
    ]
    if missing_unknown:
        print(f"[warn] time_varying_unknown_reals not in dataframe: {missing_unknown}")

    # --- SCALERS / NORMALIZERS ---
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

    # --- sample weights ---
    s = train_df["target_5d"].abs().clip(upper=train_df["target_5d"].quantile(0.99))
    train_df.loc[:, "sample_weight"] = 0.25 + 0.75 * (s / (s.median() + 1e-8))
    val_df.loc[:, "sample_weight"] = (
        1.0  # needed because training_dataset uses weight="sample_weight"
    )

    print("Creating training TimeSeriesDataSet...")
    training_dataset = TimeSeriesDataSet(
        train_df,
        **dataset_params,
        scalers=scalers,
        target_normalizer=final_target_normalizer,
        weight="sample_weight",
    )
    print("Training TimeSeriesDataSet created successfully.")

    print("Creating validation TimeSeriesDataSet from training dataset...")
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        val_df,
        allow_missing_timesteps=True,
        predict=True,
        stop_randomization=True,
    )

    print("Validation TimeSeriesDataSet created successfully.")

    if len(training_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError(
            "Train/validation split resulted in an empty dataset. Check split logic and data range."
        )
    print(
        f"Training samples: {len(training_dataset)}, Validation samples: {len(validation_dataset)}"
    )

    # ---- Stable IC evaluation over last N calendar days of the val slice ----
    last_n_days = int(OmegaConf.select(cfg, "evaluate.ic_last_n_days", default=60))
    cutoff_date = val_df["date"].max() - pd.Timedelta(days=last_n_days)
    val_ic_df = val_df[val_df["date"] >= cutoff_date].copy()

    # Create the IC callback using the dedicated loader
    rank_ic_cb = RankICCallback(
        dataset=training_dataset, target_idx=0, horizon=1, last_n_days=last_n_days
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
        lr_schedule=OmegaConf.to_container(
            cfg.trainer.get("lr_schedule", {}), resolve=True
        ),
        steps_per_epoch=int(np.ceil(len(training_dataset) / cfg.trainer.batch_size)),
        max_epochs=int(cfg.trainer.max_epochs),
    )
    print(f"Model {cfg.model._target_} (GlobalTFT wrapper) initialized.")

    # --- TRAIN DATALOADER (optional balanced sampling) ---
    num_cpu = os.cpu_count() or 8
    num_workers_cfg = int(cfg.trainer.num_workers)
    num_workers = int(min(max(0, num_workers_cfg), max(0, num_cpu - 2)))

    def _safe_prefetch_kwargs(nw: int) -> dict:
        return dict(
            persistent_workers=bool(nw > 0),
            **({"prefetch_factor": 4} if nw > 0 else {}),
        )

    train_loader_kwargs = dict(
        train=True,
        batch_size=int(cfg.trainer.batch_size),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        **_safe_prefetch_kwargs(num_workers),
    )

    use_weighted = bool(
        OmegaConf.select(cfg, "trainer.use_weighted_sampler", default=False)
    )
    train_loader = None

    # define once for sampler tail-upweighting
    main_target = target_list[0] if len(target_list) else "target_5d"

    if use_weighted:
        try:
            idx_df = training_dataset.index.copy()
            if idx_df is None or len(idx_df) == 0:
                raise RuntimeError("training_dataset.index is not available or empty")

            cols = list(idx_df.columns)

            # --- find encoded group column ---
            group_name = group_ids_list[0] if len(group_ids_list) else None
            preferred_col = f"__group_id__{group_name}" if group_name else None
            group_cols = [c for c in cols if c.startswith("__group_id__")]
            if preferred_col and preferred_col in cols:
                group_col_encoded = preferred_col
            elif group_cols:
                group_col_encoded = group_cols[0]
                print(f"[sampler] Using group column '{group_col_encoded}' (fallback).")
            elif "group_id" in cols:
                group_col_encoded = "group_id"
                print("[sampler] Using legacy 'group_id' column.")
            else:
                group_col_encoded = None  # fallback path

            # --- find time index at decoder start ---
            if "decoder_time_idx" in cols:
                time_col = "decoder_time_idx"
            elif "time_idx" in cols:
                time_col = "time_idx"
            else:
                time_candidates = [c for c in cols if c.endswith("time_idx")]
                time_col = time_candidates[0] if time_candidates else None

            if group_col_encoded is None or time_col is None:
                # Fallback – balance by sequence length when PF hides internals
                if "sequence_id" not in cols:
                    raise RuntimeError(
                        f"No encoded group/time columns and no 'sequence_id' in dataset.index. Columns: {cols[:15]}..."
                    )
                seq_counts = idx_df["sequence_id"].value_counts()
                row_weights = (
                    idx_df["sequence_id"].map(1.0 / seq_counts).astype("float64").values
                )
                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(row_weights, dtype=torch.double),
                    num_samples=len(row_weights),
                    replacement=True,
                )
                train_loader = training_dataset.to_dataloader(
                    sampler=sampler,
                    shuffle=False,
                    **train_loader_kwargs,
                )
                print(
                    f"Using sequence-balanced WeightedRandomSampler across {len(seq_counts)} sequences."
                )
            else:
                # decode ids back to original strings
                enc = training_dataset.categorical_encoders[group_name]
                encoded_vals = idx_df[group_col_encoded].to_numpy()
                try:
                    decoded_vals = enc.inverse_transform(encoded_vals)
                except Exception:
                    if hasattr(enc, "classes_"):
                        cls = enc.classes_
                        if isinstance(cls, dict):
                            inv = {v: k for k, v in cls.items()}
                            decoded_vals = np.array(
                                [str(inv.get(int(v), v)) for v in encoded_vals],
                                dtype=object,
                            )
                        else:
                            decoded_vals = np.array(
                                [
                                    str(cls[int(v)]) if int(v) < len(cls) else str(v)
                                    for v in encoded_vals
                                ],
                                dtype=object,
                            )
                    else:
                        decoded_vals = encoded_vals.astype(str)
                idx_df["ticker_id_decoded"] = decoded_vals.astype(str)
                idx_df[time_col] = idx_df[time_col].astype(np.int64)

                # base weights: 1 / sqrt(count_per_ticker)
                counts = pd.Series(idx_df["ticker_id_decoded"]).value_counts()
                base_w_map = (1.0 / np.sqrt(counts)).to_dict()
                base_w = np.array(
                    [base_w_map[v] for v in idx_df["ticker_id_decoded"]],
                    dtype=np.float64,
                )

                # tail up-weighting using main target at decoder start
                key_df = train_df[[group_name, time_idx_str, main_target]].copy()
                key_df[group_name] = key_df[group_name].astype(str)
                key_df = key_df.rename(
                    columns={
                        group_name: "ticker_id_decoded",
                        time_idx_str: time_col,
                        main_target: "y_main",
                    }
                )

                idx_df_merged = idx_df.merge(
                    key_df, on=["ticker_id_decoded", time_col], how="left"
                )
                y_abs = np.abs(idx_df_merged["y_main"].to_numpy())
                finite = np.isfinite(y_abs)
                if finite.any():
                    q70, q90 = np.nanpercentile(y_abs[finite], [70, 90])
                    tail = np.ones_like(y_abs, dtype=np.float64)
                    tail[(y_abs >= q70) & (y_abs < q90)] = 1.5
                    tail[(y_abs >= q90)] = 2.0
                    tail[~finite] = 1.0
                else:
                    tail = np.ones_like(y_abs, dtype=np.float64)

                weights = base_w * tail
                weights = np.clip(weights, 1e-6, None)

                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(weights, dtype=torch.double),
                    num_samples=len(weights),
                    replacement=True,
                )

                train_loader = training_dataset.to_dataloader(
                    train=True,
                    batch_size=cfg.trainer.batch_size,
                    sampler=sampler,
                    shuffle=False,
                    num_workers=train_loader_kwargs["num_workers"],
                    pin_memory=train_loader_kwargs["pin_memory"],
                    drop_last=train_loader_kwargs["drop_last"],
                    **_safe_prefetch_kwargs(num_workers),
                )
                print(
                    f"Using WeightedRandomSampler… groups={len(counts)}, main_target={main_target}"
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
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        **_safe_prefetch_kwargs(num_workers),
    )

    # >>> DO NOT re-instantiate RankICCallback here <<<
    # rank_ic_cb = RankICCallback(val_loader=..., ...)  # <-- removed (this caused the error)

    early_stop_callback = EarlyStopping(
        monitor="val_rank_ic",
        patience=cfg.trainer.early_stopping_patience,
        mode="max",
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(
        logging_interval=cfg.trainer.lr_monitor_logging_interval
    )

    # Save best by rank-IC (single “best” file so evaluate.py picks it)
    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        filename="{epoch}-val_rank_ic={val_rank_ic:.4f}-best",
        monitor="val_rank_ic",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    callbacks = [
        early_stop_callback,
        LearningRateMonitor(logging_interval=cfg.trainer.lr_monitor_logging_interval),
        checkpoint_callback,
        rank_ic_cb,
    ]

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
        try:
            wandb_run_dir = (
                Path(logger.experiment.dir) if hasattr(logger, "experiment") else None
            )
            if wandb_run_dir and wandb_run_dir.exists():
                hydra_cfg_path = Path(HydraConfig.get().runtime.output_dir) / ".hydra"
                target_hydra_path = wandb_run_dir / ".hydra"
                print(
                    f"Copying Hydra config from {hydra_cfg_path} to {target_hydra_path}..."
                )
                if target_hydra_path.exists():
                    shutil.rmtree(target_hydra_path)
                shutil.copytree(hydra_cfg_path, target_hydra_path)
                print("Successfully copied .hydra config directory.")
        except Exception as e:
            print(f"Config copy to W&B dir failed: {e}")

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
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 0.5),
        precision=str(cfg.trainer.get("precision", "16-mixed")),
        accumulate_grad_batches=int(cfg.trainer.get("accumulate_grad_batches", 1)),
        num_sanity_val_steps=0,
    )

    print("Trainer initialized. Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training finished.")


if __name__ == "__main__":
    main()
