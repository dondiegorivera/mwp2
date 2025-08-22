# scripts/rolling_cv.py
import os
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, ListConfig
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

# ✅ Use pytorch_lightning callbacks
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from market_prediction_workbench.model import GlobalTFT

import polars as plr  # keep polars separate from pl (pytorch_lightning)


REQUIRED_STATIC_CATS = ["ticker_id", "sector_id"]


def _load_map_df(path: Path, key: str, value: str) -> pd.DataFrame | None:
    try:
        if path.exists():
            mp = plr.read_parquet(path)
            if key in mp.columns and value in mp.columns:
                return mp.select([key, value]).to_pandas()
    except Exception:
        pass
    return None


def _ensure_pf_ready(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Make df ready for PF per fold:
      - ensure time_idx is int
      - ensure ticker_id / sector_id exist (maps -> fallback factorize)
      - cast categorical IDs to str
    """
    df = df.copy()

    # 1) time_idx dtype
    time_idx_col = getattr(cfg.data, "time_idx", "time_idx")
    if time_idx_col in df.columns:
        df[time_idx_col] = df[time_idx_col].astype(int)

    present = set(df.columns)
    missing = [c for c in REQUIRED_STATIC_CATS if c not in present]

    # 2) try rebuilding from mapping files if missing
    data_dir = Path(cfg.paths.data_dir)
    tmap_path = data_dir / "processed" / "ticker_map.parquet"
    smap_path_1 = data_dir / "processed" / "industry_map.parquet"
    smap_path_2 = data_dir / "processed" / "sector_map.parquet"

    if "ticker_id" in missing and "ticker" in df.columns:
        tmap = _load_map_df(tmap_path, "ticker", "ticker_id")
        if tmap is not None:
            df = df.merge(tmap, on="ticker", how="left")

    if "sector_id" in missing and ("industry" in df.columns or "sector" in df.columns):
        if "industry" in df.columns:
            smap = (
                _load_map_df(smap_path_1, "industry", "sector_id")
                or _load_map_df(smap_path_2, "industry", "sector_id")
            )
            if smap is not None:
                df = df.merge(smap, on="industry", how="left")
        elif "sector" in df.columns:
            smap = (
                _load_map_df(smap_path_2, "sector", "sector_id")
                or _load_map_df(smap_path_1, "sector", "sector_id")
            )
            if smap is not None:
                df = df.merge(smap, on="sector", how="left")

    # 3) fallback: factorize if still missing
    if "ticker_id" not in df.columns:
        if "ticker" in df.columns:
            codes, _ = pd.factorize(df["ticker"])
            df["ticker_id"] = codes.astype(np.int32)
        else:
            raise KeyError("Need 'ticker_id' or 'ticker' to build it; neither present.")

    if "sector_id" not in df.columns:
        key = "industry" if "industry" in df.columns else ("sector" if "sector" in df.columns else None)
        if key is not None:
            codes, _ = pd.factorize(df[key])
            df["sector_id"] = codes.astype(np.int32)
        else:
            # leave missing; we'll drop from static_categoricals for this fold
            pass

    # 4) PF expects categoricals as strings
    if "ticker_id" in df.columns:
        df["ticker_id"] = df["ticker_id"].astype(str)
    if "sector_id" in df.columns:
        df["sector_id"] = df["sector_id"].astype(str)

    return df


def _with_context_for_val(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    group_col: str,
    time_col: str,
    max_encoder_length: int,
) -> pd.DataFrame:
    """
    Add last `max_encoder_length` rows per group from train (context)
    immediately before the first validation timestamp, plus the full validation slice.
    """
    if df_val.empty:
        return df_val.copy()

    first_val_t = int(df_val[time_col].min())
    val_groups = df_val[group_col].unique()

    ctx = df_train[
        (df_train[group_col].isin(val_groups))
        & (df_train[time_col] >= first_val_t - max_encoder_length)
        & (df_train[time_col] < first_val_t)
    ].copy()

    df_val_ext = pd.concat([ctx, df_val], axis=0, ignore_index=True)
    df_val_ext.sort_values([group_col, time_col], inplace=True)

    return df_val_ext


# -------------------------- helpers --------------------------

def _cfg_list(val):
    if val is None:
        return []
    # ✅ treat OmegaConf ListConfig as a sequence
    if isinstance(val, (list, tuple, ListConfig)):
        return list(val)
    return [val]


def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise RuntimeError("Processed parquet must have a 'date' column (datetime).")
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])
    return df


def _inverse_with_groups(data: torch.Tensor, normalizer, groups: torch.Tensor) -> torch.Tensor:
    if isinstance(normalizer, GroupNormalizer):
        g = groups[:, 0].cpu().numpy()  # group id index
        scale = torch.as_tensor(normalizer.get_parameters(g), dtype=data.dtype, device=data.device)
        loc, sigm = scale[:, 0], scale[:, 1]
        while loc.dim() < data.dim():
            loc = loc.unsqueeze(1)
        while sigm.dim() < data.dim():
            sigm = sigm.unsqueeze(1)
        return data * sigm + loc
    if isinstance(normalizer, MultiNormalizer):
        parts = [
            _inverse_with_groups(data[..., i], sub, groups)
            for i, sub in enumerate(normalizer.normalizers)
        ]
        return torch.stack(parts, dim=-1)
    out = normalizer.inverse_transform(data)
    return torch.as_tensor(out, dtype=data.dtype, device=data.device)


def _build_folds(unique_dates: List[pd.Timestamp], n_folds: int, val_days: int, embargo_days: int) -> List[Dict]:
    """
    Expanding train, fixed-length validation; step = val_days.
    """
    T = len(unique_dates)
    if T < (val_days + embargo_days + 200):
        raise RuntimeError("Not enough dates to build folds with the requested settings.")

    folds = [];
    for i in range(n_folds):
        val_end_idx = T - 1 - (n_folds - 1 - i) * val_days
        val_start_idx = val_end_idx - (val_days - 1)
        train_end_idx = val_start_idx - embargo_days - 1
        if train_end_idx < 0:
            continue
        folds.append(
            dict(
                train_end=unique_dates[train_end_idx],
                val_start=unique_dates[val_start_idx],
                val_end=unique_dates[val_end_idx],
                embargo_days=embargo_days,
            )
        )
    return folds


def _make_datasets(cfg, df_train: pd.DataFrame, df_val: pd.DataFrame, target_col: str):
    # Ensure required cols & dtypes
    df_train = _ensure_pf_ready(df_train, cfg)
    df_val   = _ensure_pf_ready(df_val, cfg)

    group_id_col = cfg.data.group_ids[0]
    time_idx_col = cfg.data.time_idx

    # Sort within group by time
    df_train = df_train.sort_values([group_id_col, time_idx_col])
    df_val   = df_val.sort_values([group_id_col, time_idx_col])

    # Some folds might miss a static categorical; only keep those present in BOTH
    global_static_cats = list(getattr(cfg.data, "static_categoricals", []))
    local_static_cats = [c for c in global_static_cats if (c in df_train.columns) and (c in df_val.columns)]
    if missing := [c for c in global_static_cats if c not in local_static_cats]:
        print(f"[warn] dropping missing static_categoricals for this fold: {missing}")

    max_enc = int(cfg.data.lookback_days)
    # Allow override via +cv.min_encoder_length=XX, else sane default
    min_enc = int(getattr(cfg, "cv", {}).get("min_encoder_length", min(30, max_enc)))

    # --- TRAIN DATASET ---
    train_ds = TimeSeriesDataSet(
        df_train,
        time_idx=time_idx_col,
        target=target_col,
        group_ids=cfg.data.group_ids,

        static_categoricals=local_static_cats,
        static_reals=getattr(cfg.data, "static_reals", []),

        time_varying_known_categoricals=getattr(cfg.data, "time_varying_known_categoricals", []),
        time_varying_known_reals=getattr(cfg.data, "time_varying_known_reals", []),
        time_varying_unknown_categoricals=getattr(cfg.data, "time_varying_unknown_categoricals", []),
        time_varying_unknown_reals=getattr(cfg.data, "time_varying_unknown_reals", []),

        max_encoder_length=max_enc,
        min_encoder_length=min_enc,                        # ✅ loosen minimum
        max_prediction_length=cfg.data.max_prediction_horizon,

        target_normalizer=GroupNormalizer(groups=cfg.data.group_ids),

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,

        allow_missing_timesteps=True,                      # ✅ tolerate gaps
    )

    # --- VALIDATION DATASET WITH CONTEXT ---
    df_val_ext = _with_context_for_val(
        df_train=df_train,
        df_val=df_val,
        group_col=group_id_col,
        time_col=time_idx_col,
        max_encoder_length=max_enc,
    )

    # Use the training dataset's encoders/lengths; predict=True is fine for evaluation loaders
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        df_val_ext,
        predict=True,
        stop_randomization=True,
    )

    # Sanity guard
    if len(val_ds) == 0:
        raise RuntimeError(
            "Validation dataset ended up empty. Try lowering +cv.min_encoder_length "
            f"(now {min_enc}), increasing cv.val_days, or reducing embargo."
        )

    return train_ds, val_ds


def _loader(ds: TimeSeriesDataSet, batch_size: int, train: bool, num_workers: int) -> DataLoader:
    return ds.to_dataloader(
        train=train,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=False,
    )


def _fit_one_fold(cfg: DictConfig, train_ds: TimeSeriesDataSet, val_ds: TimeSeriesDataSet, outdir: Path) -> GlobalTFT:
    bs = int(cfg.trainer.batch_size)
    nw = int(cfg.trainer.num_workers)
    train_loader = _loader(train_ds, bs, True, nw)
    val_loader = _loader(val_ds, bs, False, nw)

    # ✅ Coerce quantiles to plain floats to avoid ListConfig issues
    q_list = [float(q) for q in _cfg_list(cfg.model.loss.quantiles)]
    loss = QuantileLoss(quantiles=q_list)

    model_params = {
        "hidden_size": int(cfg.model.hidden_size),
        "lstm_layers": int(cfg.model.lstm_layers),
        "dropout": float(cfg.model.dropout),
        "attention_head_size": int(getattr(cfg.model, "attention_head_size", 4)),
        "loss": loss,
    }

    steps_per_epoch = len(train_loader)
    model = GlobalTFT(
        model_specific_params=model_params,
        learning_rate=float(cfg.model.learning_rate),
        weight_decay=float(cfg.model.weight_decay),
        timeseries_dataset=train_ds,
        timeseries_dataset_params=None,
        lr_schedule=OmegaConf.to_container(getattr(cfg.trainer, "lr_schedule", {}), resolve=True),
        steps_per_epoch=steps_per_epoch,
        max_epochs=int(cfg.trainer.max_epochs),
    )

    # --- Callbacks ---
    callbacks = []
    if "early_stopping_patience" in cfg.trainer:
        es = EarlyStopping(
            monitor=str(cfg.trainer.early_stopping_monitor),
            patience=int(cfg.trainer.early_stopping_patience),
            mode=str(cfg.trainer.early_stopping_mode),
        )
        callbacks.append(es)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(outdir / "checkpoints"),
        filename="{epoch}-{val_loss:.2f}-best",
        save_top_k=1,
        monitor=str(cfg.trainer.early_stopping_monitor),
        mode=str(cfg.trainer.early_stopping_mode),
    )
    callbacks.append(ckpt_cb)

    # 🔕 Disable loggers for CV to avoid YAML serialization issues.
    logger = False

    # ✅ Only add LR monitor if a logger is enabled
    if logger:
        lr_mon = LearningRateMonitor(
            logging_interval=str(cfg.trainer.get("lr_monitor_logging_interval", "epoch"))
        )
        callbacks.append(lr_mon)

    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=str(cfg.trainer.accelerator),
        devices=(cfg.trainer.devices if str(cfg.trainer.devices).lower() != "auto" else "auto"),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=float(cfg.trainer.get("gradient_clip_val", 0.5)),
        precision=str(cfg.trainer.get("precision", "32-true")),
        num_sanity_val_steps=0,
        accumulate_grad_batches=int(getattr(cfg.trainer, "accumulate_grad_batches", 1)),
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    ckpt = outdir / "last.ckpt"
    trainer.save_checkpoint(str(ckpt))
    return model


@torch.no_grad()
def _predict_decode(model: GlobalTFT, ds: TimeSeriesDataSet, loader: DataLoader, target_col: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    preds_norm, trues_norm, groups_all, tickers, t_idx = [], [], [], [], []
    decoder = ds.categorical_encoders[_cfg_list(model.hparams["timeseries_dataset_params"]["group_ids"])[0]]

    for x, y in loader:
        x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
        y0 = y[0] if isinstance(y, (list, tuple)) else y
        out = model(x).prediction
        if out.dim() == 2:
            out = out.unsqueeze(-1)
        preds_norm.append(out.cpu())
        trues_norm.append(y0.cpu())
        groups_all.append(x["groups"].cpu())

        enc = x["groups"][:, 0].cpu().numpy()
        try:
            dec = decoder.inverse_transform(enc)
            tickers.extend([str(d) for d in dec])
        except Exception:
            tickers.extend([str(int(e)) for e in enc])

        t_idx.append(x["decoder_time_idx"][:, 0].cpu().numpy())

    preds_norm = torch.cat(preds_norm, dim=0)  # [B,H,Q]
    trues_norm = torch.cat(trues_norm, dim=0)  # [B,H]
    groups = torch.cat(groups_all, dim=0)      # [B,G]
    tickers = np.array(tickers)
    time_idx = np.concatenate(t_idx, axis=0)   # [B]

    normalizer = ds.target_normalizer
    Q = preds_norm.shape[-1]
    preds_dec = []
    for q in range(Q):
        preds_dec.append(_inverse_with_groups(preds_norm[..., q], normalizer, groups))
    preds_dec = torch.stack(preds_dec, dim=-1).numpy()  # [B,H,Q]
    trues_dec = _inverse_with_groups(trues_norm, normalizer, groups).numpy()  # [B,H]

    y_true = trues_dec[:, 0]
    if Q == 1:
        p50 = preds_dec[:, 0, 0]
        q_lo = p50
        q_hi = p50
    else:
        quantiles = getattr(model.model.loss, "quantiles", [0.5])
        def _nearest(qs, target):
            return int(np.argmin([abs(float(q) - target) for q in qs]))
        i50 = _nearest(quantiles, 0.5)
        ilo = _nearest(quantiles, 0.05)
        ihi = _nearest(quantiles, 0.95)
        p50 = preds_dec[:, 0, i50]
        q_lo = preds_dec[:, 0, ilo]
        q_hi = preds_dec[:, 0, ihi]

    return dict(
        ticker=tickers, time_idx=time_idx,
        y_true=y_true, p50=p50, q_lo=q_lo, q_hi=q_hi
    )


def _metrics_from_arrays(arr: Dict[str, np.ndarray]) -> Dict[str, float]:
    y = arr["y_true"]; p = arr["p50"]; lo = arr["q_lo"]; hi = arr["q_hi"]
    m = np.isfinite(y) & np.isfinite(p) & np.isfinite(lo) & np.isfinite(hi)
    y, p, lo, hi = y[m], p[m], lo[m], hi[m]
    mae = float(np.mean(np.abs(p - y)))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    cov = float(np.mean((y >= lo) & (y <= hi)))
    piw = float(np.mean(hi - lo))
    return {"mae": mae, "rmse": rmse, "coverage_90": cov, "pi_width_90": piw, "n": int(len(y))}


def _daily_rank_ic(arr: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, float, float]:
    df = pd.DataFrame({"time_idx": arr["time_idx"], "ticker": arr["ticker"], "p50": arr["p50"], "y": arr["y_true"]})
    out = []
    for t, g in df.groupby("time_idx"):
        if g["p50"].nunique() < 3 or g["y"].nunique() < 3:
            continue
        ic = spearmanr(g["p50"], g["y"]).correlation
        if np.isfinite(ic):
            out.append((t, float(ic)))
    ic_df = pd.DataFrame(out, columns=["time_idx", "rank_ic"]).sort_values("time_idx")
    return (
        ic_df,
        float(ic_df["rank_ic"].mean()) if len(ic_df) else float("nan"),
        float(ic_df["rank_ic"].std(ddof=1)) if len(ic_df) > 1 else float("nan"),
    )


def _costed_sharpe(arr: Dict[str, np.ndarray], top_q: float = 0.1, cost_bps: float = 10.0) -> float:
    """
    Simple equal-weight daily long/short using p50 rank; realized y_true (5D).
    Cost model: 1-way cost_bps applied on *turnover* (sum of absolute weight changes).
    """
    df = pd.DataFrame({"time_idx": arr["time_idx"], "ticker": arr["ticker"], "p50": arr["p50"], "y": arr["y_true"]})
    rets = []
    prev_weights = {}
    cost_rate = cost_bps * 1e-4
    for t, g in df.groupby("time_idx"):
        g = g.sort_values("p50")
        n = len(g)
        k = max(1, int(n * top_q))
        short = g.iloc[:k]
        long = g.iloc[-k:]
        w = {**{tic: -1.0 / k for tic in short["ticker"]},
             **{tic:  1.0 / k for tic in long["ticker"]}}
        keys = set(w.keys()) | set(prev_weights.keys())
        turnover = sum(abs(w.get(k_, 0.0) - prev_weights.get(k_, 0.0)) for k_ in keys)
        gross = float(long["y"].mean() - short["y"].mean())
        net = gross - cost_rate * turnover
        rets.append(net)
        prev_weights = w
    r = np.array(rets, dtype=float)
    if len(r) < 2 or np.allclose(r.std(ddof=1), 0.0):
        return float("nan")
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252.0))
    return sharpe


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Silence Tensor Core precision warning and speed up matmuls
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    # Defaults for CV (override via +cv.*)
    cv = dict(n_folds=5, val_days=120, embargo_days=int(cfg.data.split.embargo_days))
    if hasattr(cfg, "cv"):
        cv.update({k: int(v) for k, v in OmegaConf.to_container(cfg.cv, resolve=True).items() if k in cv})

    print(f"=== Rolling CV settings === {cv}")

    parquet_path = Path(cfg.paths.processed_data_file)
    if not parquet_path.exists():
        raise SystemExit(f"Processed parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = _ensure_dt(df)

    # Phase 2 scope: 5-day only
    targets = _cfg_list(cfg.data.target)
    if len(targets) != 1 or targets[0] != "target_5d":
        print(f"[info] forcing target to 'target_5d' for CV (was {targets})")
    target_col = "target_5d"

    dates = sorted(df["date"].unique().tolist())
    folds = _build_folds(dates, cv["n_folds"], cv["val_days"], cv["embargo_days"])
    if not folds:
        raise SystemExit("No valid folds constructed. Adjust val_days / embargo_days.")
    print("Folds:")
    for i, f in enumerate(folds, 1):
        print(f"  Fold {i}: train_end ≤ {f['train_end'].date()} | val ∈ [{f['val_start'].date()}, {f['val_end'].date()}] | embargo {f['embargo_days']}d")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_root = Path(cfg.paths.log_dir) / "cv" / ts
    out_root.mkdir(parents=True, exist_ok=True)

    per_fold = []
    for i, f in enumerate(folds, 1):
        print(f"\n=== Fold {i}/{len(folds)} ===")
        d_te, d_vs, d_ve = f["train_end"], f["val_start"], f["val_end"]

        df_train = df[df["date"] <= d_te].copy()
        df_val = df[(df["date"] >= d_vs) & (df["date"] <= d_ve)].copy()

        df_train[cfg.data.time_idx] = df_train[cfg.data.time_idx].astype(int)
        df_val[cfg.data.time_idx] = df_val[cfg.data.time_idx].astype(int)

        train_ds, val_ds = _make_datasets(cfg, df_train, df_val, target_col)

        fold_dir = out_root / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model = _fit_one_fold(cfg, train_ds, val_ds, fold_dir)

        val_loader = _loader(val_ds, int(cfg.trainer.batch_size), False, int(cfg.trainer.num_workers))
        arr = _predict_decode(model, train_ds, val_loader, target_col)

        m = _metrics_from_arrays(arr)
        ic_df, ic_mean, ic_std = _daily_rank_ic(arr)
        sharpe = _costed_sharpe(arr, top_q=0.1, cost_bps=10.0)

        m["rank_ic_mean"] = float(ic_mean)
        m["rank_ic_std"] = float(ic_std)
        m["costed_sharpe_ls10"] = float(sharpe)

        ic_df.to_csv(fold_dir / "daily_ic.csv", index=False)
        pd.DataFrame(arr).head(5000).to_csv(fold_dir / "predictions_sample.csv", index=False)
        with open(fold_dir / "metrics.json", "w") as fp:
            json.dump(m, fp, indent=2)
        per_fold.append(m)

        print("Fold metrics:", json.dumps(m, indent=2))

    agg = {}
    if per_fold:
        keys = per_fold[0].keys()
        for k in keys:
            vals = [pf[k] for pf in per_fold if isinstance(pf[k], (int, float)) and np.isfinite(pf[k])]
            if not vals:
                continue
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    with open(out_root / "summary.json", "w") as fp:
        json.dump(agg, fp, indent=2)
    print("\n=== CV summary ===")
    print(json.dumps(agg, indent=2))
    print(f"\nArtifacts: {out_root}")


if __name__ == "__main__":
    # safer defaults for CUDA alloc to reduce fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
