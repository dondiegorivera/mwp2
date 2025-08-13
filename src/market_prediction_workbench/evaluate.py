# src/market_prediction_workbench/evaluate.py
import json
import sys
import inspect
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from market_prediction_workbench.model import GlobalTFT


# -----------------------------------------------------------------------------#
# Helpers                                                                       #
# -----------------------------------------------------------------------------#
def _cfg_list(val) -> List[str]:
    if val is None:
        return []
    if isinstance(val, (str, int, float)):
        return [str(val)]
    if isinstance(val, (list, ListConfig)):
        return [str(v) for v in val]
    raise TypeError(f"Unsupported cfg node type: {type(val)}")


def _safe_parse_val_loss(stem: str) -> float:
    if "val_loss=" not in stem:
        return float("inf")
    try:
        return float(stem.split("val_loss=")[-1].split("-")[0])
    except Exception:
        return float("inf")


def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _inverse_with_groups(
    data: torch.Tensor, normalizer, groups: torch.Tensor
) -> torch.Tensor:
    """
    Robust inverse transform that works across PF versions:
      • GroupNormalizer (any transform)
      • MultiNormalizer (recurses)
      • Others (delegate)
    """
    # GroupNormalizer
    if isinstance(normalizer, GroupNormalizer):
        sig = inspect.signature(normalizer.inverse_transform)
        if "group_ids" in sig.parameters:
            kw = "group_ids"
        elif "groups" in sig.parameters:
            kw = "groups"
        elif "target_scale" in sig.parameters:
            kw = "target_scale"
        elif "scale" in sig.parameters:
            kw = "scale"
        else:
            kw = None

        g = groups[:, 0].cpu().numpy()  # [B]
        scale = torch.as_tensor(
            normalizer.get_parameters(g),  # [B, 2] (loc, scale)
            dtype=data.dtype,
            device=data.device,
        )

        try:
            if kw is None:
                return normalizer.inverse_transform(data, scale)
            else:
                return normalizer.inverse_transform(data, **{kw: scale})
        except NotImplementedError:
            loc = scale[:, 0]
            sigm = scale[:, 1]
            while loc.dim() < data.dim():
                loc = loc.unsqueeze(1)
            while sigm.dim() < data.dim():
                sigm = sigm.unsqueeze(1)
            return data * sigm + loc

    # MultiNormalizer
    if isinstance(normalizer, MultiNormalizer):
        parts = [
            _inverse_with_groups(data[..., i], sub, groups)
            for i, sub in enumerate(normalizer.normalizers)
        ]
        return torch.stack(parts, dim=-1)

    # Others
    out = normalizer.inverse_transform(data)
    return torch.as_tensor(out, dtype=data.dtype, device=data.device)


# -----------------------------------------------------------------------------#
# Inference                                                                    #
# -----------------------------------------------------------------------------#
def run_inference(
    model: GlobalTFT, loader: DataLoader, cfg: DictConfig, dataset: TimeSeriesDataSet
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str], int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    group_id_col = _cfg_list(cfg.data.group_ids)[0]
    decoder = dataset.categorical_encoders[group_id_col]

    # Build a robust inverse map for tickers (works even if decoder.inverse_transform complains)
    inv_map = None
    if hasattr(decoder, "classes_"):
        cls = decoder.classes_
        if isinstance(cls, dict):
            inv_map = {v: k for k, v in cls.items()}  # value->idx to idx->value
        else:
            # list/array-like where positions are the encoded ints
            inv_map = {i: str(v) for i, v in enumerate(list(cls))}

    preds_norm_list, trues_norm_list = [], []
    tickers_all, t_idx_all = [], []
    groups_all = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Running inference"):
            x = _move_to_device(x, device)

            target_norm = y[0] if isinstance(y, (list, tuple)) else y
            output_norm = model(x).prediction  # [B,H,Q] or [B,H,T*Q] depending on PF

            # Ensure [B,H,T,Q]
            if isinstance(output_norm, list):
                output_norm = torch.stack(output_norm, dim=2)
            else:
                output_norm = output_norm.unsqueeze(2)

            if isinstance(target_norm, list):
                target_norm = torch.stack(target_norm, dim=2)
            else:
                target_norm = target_norm.unsqueeze(2)

            preds_norm_list.append(output_norm.cpu())
            trues_norm_list.append(target_norm.cpu())

            groups_all.append(x["groups"].cpu())

            encoded = x["groups"][:, 0].cpu().numpy()
            # Try decoder.inverse_transform; fall back to inv_map; last resort = encoded int as string
            try:
                decoded_vals = decoder.inverse_transform(encoded)
                tickers_all.extend(
                    str(int(s)) if str(s).isdigit() else str(s) for s in decoded_vals
                )
            except Exception:
                if inv_map is not None:
                    tickers_all.extend(str(inv_map.get(int(e), "UNK")) for e in encoded)
                else:
                    tickers_all.extend(str(int(e)) for e in encoded)

            t_idx_all.append(x["decoder_time_idx"].cpu().numpy())

    preds_norm = torch.cat(preds_norm_list, dim=0)  # [B, H, T, Q]
    trues_norm = torch.cat(trues_norm_list, dim=0)  # [B, H, T]
    groups = torch.cat(groups_all, dim=0)  # [B, G]
    tickers = np.array(tickers_all)
    time_idx = np.concatenate(t_idx_all, axis=0)  # [B, H]

    normalizer = dataset.target_normalizer

    # Decode predictions per target & quantile
    preds_dec = []
    num_targets = preds_norm.shape[2]
    num_horizons = preds_norm.shape[1]
    num_quantiles = preds_norm.shape[3]

    for i in range(num_targets):
        target_preds_by_q = []
        for q in range(num_quantiles):
            pred_data = preds_norm[:, :, i, q]
            if isinstance(normalizer, MultiNormalizer) and i < len(
                normalizer.normalizers
            ):
                norm = normalizer.normalizers[i]
            else:
                norm = normalizer
            decoded = _inverse_with_groups(pred_data, norm, groups)
            target_preds_by_q.append(decoded)
        preds_dec.append(torch.stack(target_preds_by_q, dim=-1))  # [B,H,Q]
    preds_dec = torch.stack(preds_dec, dim=2)  # [B,H,T,Q]

    trues_dec = _inverse_with_groups(trues_norm, normalizer, groups)  # [B,H,T]

    pred_dict, true_dict = {}, {"ticker": tickers}
    short_names = [t.replace("target_", "") for t in _cfg_list(cfg.data.target)]

    # Keep h1 time index for plotting
    true_dict["time_idx_h1"] = time_idx[:, 0]

    for i, name in enumerate(short_names):
        for h in range(num_horizons):
            suffix = f"@h{h+1}"
            pred_dict[f"{name}_lower{suffix}"] = preds_dec[:, h, i, 0].numpy()
            pred_dict[f"{name}{suffix}"] = preds_dec[:, h, i, 1].numpy()
            pred_dict[f"{name}_upper{suffix}"] = preds_dec[:, h, i, 2].numpy()
            true_dict[f"{name}{suffix}"] = trues_dec[:, h, i].numpy()

    return pred_dict, true_dict, short_names, num_horizons


# -----------------------------------------------------------------------------#
# Evaluation Metrics                                                           #
# -----------------------------------------------------------------------------#
def evaluate(preds: Dict, trues: Dict, short_names: List[str]) -> Dict[str, float]:
    metrics = {}

    # infer horizons from keys like "<name>@hX"
    horizons = set()
    for k in preds.keys():
        if "@h" in k and not k.endswith("_lower") and not k.endswith("_upper"):
            try:
                horizons.add(int(k.split("@h")[-1]))
            except Exception:
                continue
    horizons = sorted(horizons)

    for name in short_names:
        for h in horizons:
            base = f"{name}@h{h}"
            if base not in preds or base not in trues:
                continue

            pred = preds[base]
            true = trues[base]

            mask = ~(np.isnan(pred) | np.isnan(true))
            pred = pred[mask]
            true = true[mask]
            if len(pred) == 0:
                continue

            mae = np.mean(np.abs(pred - true))
            rmse = np.sqrt(np.mean((pred - true) ** 2))

            lower = preds.get(f"{name}_lower@h{h}", np.full_like(pred, np.nan))[mask]
            upper = preds.get(f"{name}_upper@h{h}", np.full_like(pred, np.nan))[mask]
            coverage = np.mean((true >= lower) & (true <= upper))

            metrics[f"{name}_mae@h{h}"] = float(mae)
            metrics[f"{name}_rmse@h{h}"] = float(rmse)
            metrics[f"{name}_coverage_90@h{h}"] = float(coverage)

    return metrics


# -----------------------------------------------------------------------------#
# Visualisation & Output                                                       #
# -----------------------------------------------------------------------------#
def _safe_ticker_id(df: pl.DataFrame, ticker: str) -> int | None:
    try:
        return df.filter(pl.col("ticker") == ticker)["ticker_id"].item()
    except Exception:
        return None


def plot_preds(preds, trues, out_dir, ticker_map, sample_tickers, short_tgt_names):
    out_dir.mkdir(parents=True, exist_ok=True)
    ticker_csv_dir = out_dir / "ticker_predictions_for_plot"
    ticker_csv_dir.mkdir(parents=True, exist_ok=True)

    # Plot first horizon (h1)
    df_plot = pd.DataFrame(
        {"ticker": trues["ticker"], "time_idx": trues["time_idx_h1"]}
    )
    # 👉 ensure string type so we can compare to ticker_map IDs reliably
    df_plot["ticker"] = df_plot["ticker"].astype(str)

    for name in short_tgt_names:
        df_plot[name] = trues.get(f"{name}@h1")
        df_plot[f"p_{name}"] = preds.get(f"{name}@h1")
        df_plot[f"p_{name}_lower"] = preds.get(
            f"{name}_lower_cal@h1", preds.get(f"{name}_lower@h1")
        )
        df_plot[f"p_{name}_upper"] = preds.get(
            f"{name}_upper_cal@h1", preds.get(f"{name}_upper@h1")
        )

    for tk in sample_tickers:
        tid = _safe_ticker_id(ticker_map, tk)
        if tid is None:
            print(f"'{tk}' not found in ticker_map – skipping.")
            continue

        # 👉 compare as strings
        tid_str = str(tid)
        df_ticker = df_plot[df_plot["ticker"] == tid_str].sort_values("time_idx")

        ticker_csv_path = ticker_csv_dir / f"{tk}_predictions.csv"
        df_ticker.to_csv(ticker_csv_path, index=False)
        print(f"Saved prediction data for '{tk}' to {ticker_csv_path}")

        if len(df_ticker) < 2:
            print(f"Ticker '{tk}' has <2 predictions – skipping plot.")
            continue

        plt.figure(figsize=(15, 4 * len(short_tgt_names)))
        for i, name in enumerate(short_tgt_names):
            plt.subplot(len(short_tgt_names), 1, i + 1)
            plt.plot(df_ticker["time_idx"], df_ticker[name], "b-", label="Actual")
            plt.plot(
                df_ticker["time_idx"],
                df_ticker[f"p_{name}"],
                "r--",
                label="Pred (median)",
            )
            plt.fill_between(
                df_ticker["time_idx"],
                df_ticker[f"p_{name}_lower"],
                df_ticker[f"p_{name}_upper"],
                alpha=0.2,
                color="orange",
                label="90% PI",
            )
            plt.title(f"{tk} – {name.upper()} horizon (h1)")
            plt.ylabel("Log-return")
            plt.legend()

        plt.xlabel("Time index")
        plt.tight_layout()
        plt.savefig(out_dir / f"{tk}_timeseries.png")
        plt.close()
        print(f"Successfully generated plot for {tk}.")


def save_output(
    metrics: Dict,
    preds: Dict,
    trues: Dict,
    out_dir: Path,
    ticker_map: pl.DataFrame | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w") as fp:
        json.dump({k: float(v) for k, v in metrics.items()}, fp, indent=2)

    df = pd.DataFrame({**trues, **preds})

    # Optional: add human-readable symbol
    if (
        ticker_map is not None
        and "ticker_id" in ticker_map.columns
        and "ticker" in ticker_map.columns
    ):
        # build id->symbol map (ids as strings to match our 'ticker' column)
        id_to_symbol = dict(
            zip(
                ticker_map["ticker_id"].cast(pl.Utf8).to_list(),
                ticker_map["ticker"].to_list(),
            )
        )
        df["ticker_symbol"] = df["ticker"].astype(str).map(id_to_symbol).fillna("UNK")

    df.to_csv(out_dir / "predictions.csv", index=False)
    print(f"\nResults written to {out_dir}")


def compute_calibration_alphas(
    preds: Dict, trues: Dict, short_names: List[str]
) -> Dict[str, float]:
    """
    Compute per-target alpha from horizon 1 only:
      alpha = 90th percentile of |resid| / (1.645 * s_hat),
      where s_hat = (upper - lower) / (2 * 1.645)
    """
    alphas = {}
    for name in short_names:
        p50 = preds.get(f"{name}@h1")
        y = trues.get(f"{name}@h1")
        lo = preds.get(f"{name}_lower@h1")
        hi = preds.get(f"{name}_upper@h1")

        if p50 is None or y is None or lo is None or hi is None:
            alphas[name] = float("nan")
            continue

        s_hat = (hi - lo) / (2.0 * 1.645)
        denom = 1.645 * s_hat
        resid = y - p50

        mask = np.isfinite(resid) & np.isfinite(denom) & (denom > 0)
        if not np.any(mask):
            alphas[name] = float("nan")
            continue

        k = np.abs(resid[mask]) / denom[mask]
        alphas[name] = float(np.nanpercentile(k, 90))
    return alphas


def add_calibrated_intervals(
    preds: Dict, short_names: List[str], num_horizons: int, alphas: Dict[str, float]
) -> None:
    """
    Add calibrated intervals as NEW keys:
      *_lower_cal@h*, *_upper_cal@h*
    (Original *_lower@h*/*_upper@h* are left untouched.)
    """
    for name in short_names:
        alpha = alphas.get(name, float("nan"))
        for h in range(1, num_horizons + 1):
            key_mid = f"{name}@h{h}"
            key_lo = f"{name}_lower@h{h}"
            key_hi = f"{name}_upper@h{h}"
            if key_mid not in preds or key_lo not in preds or key_hi not in preds:
                continue

            mid = preds[key_mid]
            lo = preds[key_lo]
            hi = preds[key_hi]

            s_hat = (hi - lo) / (2.0 * 1.645)

            if np.isnan(alpha) or alpha <= 0:
                lo_cal = lo
                hi_cal = hi
            else:
                lo_cal = mid - 1.645 * alpha * s_hat
                hi_cal = mid + 1.645 * alpha * s_hat

            preds[f"{name}_lower_cal@h{h}"] = lo_cal
            preds[f"{name}_upper_cal@h{h}"] = hi_cal


def evaluate_with_calibrated(
    preds: Dict, trues: Dict, short_names: List[str], num_horizons: int
) -> Dict[str, float]:
    """
    Reuse the existing `evaluate()` but temporarily point lower/upper
    to the calibrated versions for coverage computation.
    """
    cal_preds = dict(preds)  # shallow copy
    for name in short_names:
        for h in range(1, num_horizons + 1):
            lo_cal = cal_preds.get(f"{name}_lower_cal@h{h}")
            hi_cal = cal_preds.get(f"{name}_upper_cal@h{h}")
            if lo_cal is not None and hi_cal is not None:
                cal_preds[f"{name}_lower@h{h}"] = lo_cal
                cal_preds[f"{name}_upper@h{h}"] = hi_cal
    return evaluate(cal_preds, trues, short_names)


# -----------------------------------------------------------------------------#
# Entry-point                                                                  #
# -----------------------------------------------------------------------------#
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    log_dir = Path(cfg.paths.log_dir)

    ckpts = list(log_dir.glob("**/*.ckpt"))
    if not ckpts:
        sys.exit(f"No .ckpt files found under {log_dir}")

    run_dirs = {
        p.parent.parent if p.parent.name == "checkpoints" else p.parent for p in ckpts
    }
    latest_run = max(
        run_dirs, key=lambda d: max(x.stat().st_mtime for x in d.glob("**/*.ckpt"))
    )
    print(f"Identified latest run directory: {latest_run}")

    cand_ckpts = list((latest_run / "checkpoints").glob("*.ckpt")) or list(
        latest_run.glob("*.ckpt")
    )
    cand_ckpts.sort(key=lambda p: _safe_parse_val_loss(p.stem))
    best = next((p for p in cand_ckpts if "best" in p.stem), None) or cand_ckpts[0]
    print(f"\nEvaluating checkpoint: {best}")

    parquet_path = Path(cfg.paths.processed_data_file)
    if not parquet_path.exists():
        sys.exit(f"Processed data file not found at {parquet_path}")

    print("Loading full processed dataset to match model architecture...")
    df_full = pd.read_parquet(parquet_path)

    # load dataset parameters from checkpoint
    cp = torch.load(best, map_location="cpu", weights_only=False)
    ds_params = cp["hyper_parameters"]["timeseries_dataset_params"]

    # ensure dtypes are right for encoders
    time_idx_col = ds_params["time_idx"]
    df_full[time_idx_col] = df_full[time_idx_col].astype(int)
    all_cat_cols = (
        ds_params.get("static_categoricals", [])
        + ds_params.get("time_varying_known_categoricals", [])
        + ds_params.get("time_varying_unknown_categoricals", [])
    )
    for col in all_cat_cols:
        if col in df_full.columns:
            df_full[col] = df_full[col].astype(str)
    print(f"Ensured '{time_idx_col}' is int and categoricals are strings.")

    # Make encoders tolerate unknown categories (add_nan=True) – helps avoid crashes in transforms
    if "categorical_encoders" in ds_params and isinstance(
        ds_params["categorical_encoders"], dict
    ):
        for name, enc in ds_params["categorical_encoders"].items():
            if hasattr(enc, "add_nan") and getattr(enc, "add_nan") is False:
                enc.add_nan = True

    # Build the full training-compatible dataset to get encoders and normalizers
    full_dataset = TimeSeriesDataSet.from_parameters(ds_params, df_full, predict=False)
    print(f"Recreated full dataset with {len(full_dataset)} samples.")

    # Filter evaluation data to ONLY tickers known by the trained encoder to avoid 'unknown' issues
    group_id_col = _cfg_list(ds_params.get("group_ids", []))[0]
    enc = full_dataset.categorical_encoders[group_id_col]
    known_values: set[str]
    if hasattr(enc, "classes_"):
        if isinstance(enc.classes_, dict):
            known_values = set(str(k) for k in enc.classes_.keys())
        else:
            known_values = set(str(v) for v in list(enc.classes_))
    else:
        known_values = set()

    # Decide evaluation scope from config
    want = _cfg_list(cfg.evaluate.sample_tickers)
    evaluate_all = any(t.upper() in {"ALL", "*"} for t in want)

    ticker_map_path = Path(cfg.paths.data_dir) / "processed" / "ticker_map.parquet"
    ticker_map = pl.read_parquet(ticker_map_path) if ticker_map_path.exists() else None
    if ticker_map is None:
        sys.exit("ticker_map.parquet not found. Cannot map ticker names.")

    if evaluate_all:
        eval_df = df_full[df_full[group_id_col].astype(str).isin(known_values)].copy()
        dropped = len(df_full) - len(eval_df)
        if dropped > 0:
            print(
                f"Filtered out {dropped} rows with unseen {group_id_col} (kept {len(eval_df)})."
            )
        # choose some tickers just for plotting (first 3 by default)
        sample_tickers_for_plots = ticker_map.select("ticker").to_series().to_list()[:3]
    else:
        filtered_map = ticker_map.filter(pl.col("ticker").is_in(want))
        ticker_ids_to_evaluate = [str(i) for i in filtered_map["ticker_id"].to_list()]
        # intersect with known
        keep_ids = [tid for tid in ticker_ids_to_evaluate if tid in known_values]
        if not keep_ids:
            sys.exit(
                f"None of {want} are in the trained encoder's known {group_id_col} set."
            )
        eval_df = df_full[df_full[group_id_col].astype(str).isin(set(keep_ids))].copy()
        sample_tickers_for_plots = want

    # Build eval dataset from filtered DataFrame using the SAME parameters
    eval_dataset = TimeSeriesDataSet.from_parameters(ds_params, eval_df, predict=False)
    print(f"Evaluation dataset has {len(eval_dataset)} samples after filtering.")

    # load model
    model = GlobalTFT.load_from_checkpoint(
        best, timeseries_dataset=full_dataset, map_location="cpu"
    )
    print("Model loaded successfully.")

    # dataloader
    loader = eval_dataset.to_dataloader(
        train=False,
        batch_size=cfg.evaluate.batch_size,
        shuffle=False,
        num_workers=cfg.evaluate.num_workers,
    )
    print("Created dataloader for evaluation.")

    preds, trues, short_tgt_names, num_horizons = run_inference(
        model, loader, cfg, full_dataset
    )
    metrics = evaluate(preds, trues, short_tgt_names)

    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Coverage table
    def _horizons_from(preds_dict):
        hs = set()
        for k in preds_dict:
            if "@h" in k and not k.endswith("_lower") and not k.endswith("_upper"):
                try:
                    hs.add(int(k.split("@h")[-1]))
                except Exception:
                    pass
        return sorted(hs) or [1]

    horizons = _horizons_from(preds)
    print("\nCoverage@90% by target & horizon")
    print("target,horizon,n,coverage_90")
    for name in short_tgt_names:
        for h in horizons:
            base = f"{name}@h{h}"
            if base not in preds or base not in trues:
                continue
            y = trues[base]
            yhat_l = preds.get(f"{name}_lower@h{h}")
            yhat_u = preds.get(f"{name}_upper@h{h}")
            m = ~(np.isnan(y) | np.isnan(yhat_l) | np.isnan(yhat_u))
            n = int(m.sum())
            cov = (
                float(np.mean((y[m] >= yhat_l[m]) & (y[m] <= yhat_u[m])))
                if n > 0
                else float("nan")
            )
            print(f"{name},{h},{n},{cov:.3f}")

    out_dir = Path(cfg.paths.log_dir) / "evaluation" / latest_run.name

    # ---- Calibration (adds *_lower_cal@h*, *_upper_cal@h*; originals untouched) ----
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = compute_calibration_alphas(preds, trues, short_tgt_names)
    print("\n--- Calibration α (per target, from h1) ---")
    for k, v in alphas.items():
        if np.isfinite(v):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: NaN")

    # add calibrated intervals for all horizons
    add_calibrated_intervals(preds, short_tgt_names, num_horizons, alphas)

    # evaluate coverage using calibrated bands
    metrics_cal = evaluate_with_calibrated(preds, trues, short_tgt_names, num_horizons)

    # write extra artifacts
    with (out_dir / "metrics_calibrated.json").open("w") as fp:
        json.dump({k: float(v) for k, v in metrics_cal.items()}, fp, indent=2)
    with (out_dir / "calibration_alphas.json").open("w") as fp:
        json.dump(alphas, fp, indent=2)

    plot_preds(
        preds, trues, out_dir, ticker_map, sample_tickers_for_plots, short_tgt_names
    )
    save_output(metrics, preds, trues, out_dir, ticker_map)


if __name__ == "__main__":
    main()
