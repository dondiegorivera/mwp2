# src/market_prediction_workbench/evaluate.py
import json
import sys
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Union

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
    """Return a list of strings from a Hydra list / scalar / None."""
    if val is None:
        return []
    if isinstance(val, (str, int, float)):
        return [str(val)]
    if isinstance(val, (list, ListConfig)):
        return [str(v) for v in val]
    raise TypeError(f"Unsupported cfg node type: {type(val)}")


def _safe_parse_val_loss(stem: str) -> float:
    """
    Extract `val_loss=<float>` from a checkpoint stem.
    Returns +inf when the pattern is missing or malformed so that `.sort()` can
    still work without crashing.
    """
    if "val_loss=" not in stem:
        return float("inf")
    try:
        return float(stem.split("val_loss=")[-1].split("-")[0])
    except Exception:
        return float("inf")


def _move_to_device(obj, device):
    """Recursively send tensors to the selected device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


# evaluate.py
# -----------------------------------------------------------------------
# REPLACE the whole inverse_transform_with_groups(...) helper with this
# -----------------------------------------------------------------------
from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer

# evaluate.py  – replace _inverse_with_groups with this version
# evaluate.py  ------------------------------------------------------------
import inspect
import torch
from pytorch_forecasting.data.encoders import GroupNormalizer, MultiNormalizer

def _inverse_with_groups(data: torch.Tensor,
                         normalizer,
                         groups: torch.Tensor) -> torch.Tensor:
    """
    Robust inverse transform that works on every PF version:
       • GroupNormalizer        (any transformation / any release)
       • MultiNormalizer        (recursion)
       • Other normalisers      (delegate)
    """
    # ------------------------------------------------------------------ #
    # 1) GroupNormalizer                                                #
    # ------------------------------------------------------------------ #
    if isinstance(normalizer, GroupNormalizer):
        # figure out which keyword, if any, the current build understands
        sig = inspect.signature(normalizer.inverse_transform)
        if   "group_ids"    in sig.parameters: kw = "group_ids"
        elif "groups"       in sig.parameters: kw = "groups"
        elif "target_scale" in sig.parameters: kw = "target_scale"
        elif "scale"        in sig.parameters: kw = "scale"
        else:                                 kw = None   # positional only

        # obtain µ (location) and σ (scale) for every sample in the batch
        g = groups[:, 0].cpu().numpy()                    # [B]
        scale = torch.as_tensor(
            normalizer.get_parameters(g),                 # [B, 2] (loc, scale)
            dtype=data.dtype,
            device=data.device,
        )

        # try the built-in inverse first …
        try:
            if kw is None:
                return normalizer.inverse_transform(data, scale)
            else:
                return normalizer.inverse_transform(data, **{kw: scale})

        # … fall back to manual µ+σ·ŷ when NotImplementedError is raised
        except NotImplementedError:
            loc  = scale[:, 0]
            sigm = scale[:, 1]
            while loc.dim()  < data.dim(): loc  = loc.unsqueeze(1)
            while sigm.dim() < data.dim(): sigm = sigm.unsqueeze(1)
            return data * sigm + loc

    # ------------------------------------------------------------------ #
    # 2) MultiNormalizer                                                 #
    # ------------------------------------------------------------------ #
    if isinstance(normalizer, MultiNormalizer):
        parts = [
            _inverse_with_groups(data[..., i], sub, groups)
            for i, sub in enumerate(normalizer.normalizers)
        ]
        return torch.stack(parts, dim=-1)

    # ------------------------------------------------------------------ #
    # 3) Anything else                                                   #
    # ------------------------------------------------------------------ #
    out = normalizer.inverse_transform(data)
    return torch.as_tensor(out, dtype=data.dtype, device=data.device)




def inverse_transform_with_groups(
    data: torch.Tensor, normalizer, groups: torch.Tensor
) -> torch.Tensor:
    """
    Inverse-transform a tensor normalized by:
      - GroupNormalizer  => manually invert using get_parameters
      - MultiNormalizer  => recurse into each sub-normalizer
      - others           => call built-in inverse_transform
    """
    # Handle GroupNormalizer
    if isinstance(normalizer, GroupNormalizer):
        group_ids = groups[:, 0].cpu().numpy()
        params = normalizer.get_parameters(group_ids)
        mus = torch.from_numpy(params[:, 0]).unsqueeze(1).to(data.device)
        return (data + 1) * mus           

    # Handle MultiNormalizer
    if isinstance(normalizer, MultiNormalizer):
        # Special case for single-element data (single target)
        if data.dim() == 1 or (data.dim() == 2 and data.shape[1] == 1):
            # Process first target only
            return inverse_transform_with_groups(
                data, normalizer.normalizers[0], groups
            )

        # Multi-target case - process each target separately
        parts = [
            inverse_transform_with_groups(data[..., i], subnorm, groups)
            for i, subnorm in enumerate(normalizer.normalizers)
        ]
        return torch.stack(parts, dim=-1)

    # Handle other normalizers
    arr = normalizer.inverse_transform(data.cpu().numpy())
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    return arr.to(data.device)


# -----------------------------------------------------------------------------#
# Inference                                                                    #
# -----------------------------------------------------------------------------#


def run_inference(
    model: GlobalTFT, loader: DataLoader, cfg: DictConfig, dataset: TimeSeriesDataSet
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Run model on the dataloader and collect predictions with inverse transform."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    group_id_col = _cfg_list(cfg.data.group_ids)[0]
    decoder = dataset.categorical_encoders[group_id_col]

    preds_norm_list, trues_norm_list = [], []
    tickers_all, t_idx_all = [], []
    groups_all = []  # Store groups for inverse transform

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Running inference"):
            x = _move_to_device(x, device)

            # normalized target
            if isinstance(y, (list, tuple)):
                target_norm = y[0]
            else:
                target_norm = y

            # forward
            output_norm = model(x).prediction
            if isinstance(output_norm, list):
                output_norm = torch.stack(output_norm, dim=2)
            else:
                output_norm = output_norm.unsqueeze(2)

            if isinstance(target_norm, list):
                target_norm = torch.stack(target_norm, dim=2)
            else:
                target_norm = target_norm.unsqueeze(2)

            # collect on CPU
            preds_norm_list.append(output_norm.cpu())
            trues_norm_list.append(target_norm.cpu())

            # Store groups for later use in inverse transform
            groups_all.append(x["groups"].cpu())

            # decode tickers & times
            encoded = x["groups"][:, 0].cpu().numpy()
            tickers_all.extend(int(s) for s in decoder.inverse_transform(encoded))
            t_idx_all.append(x["decoder_time_idx"].cpu().numpy())

    # concatenate
    preds_norm = torch.cat(preds_norm_list, dim=0)  # [B, H, T, Q]
    trues_norm = torch.cat(trues_norm_list, dim=0)  # [B, H, T]
    groups = torch.cat(groups_all, dim=0)  # [B, G] where G is number of group columns
    tickers = np.array(tickers_all)
    time_idx = np.concatenate(t_idx_all, axis=0)  # [B, H]

    # Handle different types of normalizers
    normalizer = dataset.target_normalizer

    # Inverse-transform predictions
    preds_dec = []
    num_targets = preds_norm.shape[2]

    for i in range(num_targets):
        target_preds = []
        for q in range(preds_norm.shape[3]):
            pred_data = preds_norm[:, :, i, q]

            # Get the appropriate normalizer
            if isinstance(normalizer, MultiNormalizer) and i < len(
                normalizer.normalizers
            ):
                norm = normalizer.normalizers[i]
            else:
                norm = normalizer

            # Handle GroupNormalizer specifically
            if isinstance(norm, GroupNormalizer):
                decoded = _inverse_with_groups(pred_data, norm, groups)
            else:
                decoded = norm.inverse_transform(pred_data)
                if not isinstance(decoded, torch.Tensor):
                    decoded = torch.tensor(decoded)

            target_preds.append(decoded)

        preds_dec.append(torch.stack(target_preds, dim=-1))

    # Stack targets: [B, H, T, Q]
    preds_dec = torch.stack(preds_dec, dim=2)

    # Inverse-transform true values
    trues_dec = _inverse_with_groups(trues_norm, normalizer, groups)

    # Extract first horizon predictions
    preds_h1 = preds_dec[:, 0]  # [B, T, Q]
    trues_h1 = trues_dec[:, 0]  # [B, T]

    # Build result dictionaries
    pred_dict, true_dict = {}, {"ticker": tickers, "time_idx": time_idx[:, 0]}
    short_names = [t.replace("target_", "") for t in _cfg_list(cfg.data.target)]

    for i, name in enumerate(short_names):
        pred_dict[f"{name}_lower"] = preds_h1[:, i, 0].numpy()
        pred_dict[f"{name}"] = preds_h1[:, i, 1].numpy()
        pred_dict[f"{name}_upper"] = preds_h1[:, i, 2].numpy()
        true_dict[name] = trues_h1[:, i].numpy()

    return pred_dict, true_dict, short_names


# -----------------------------------------------------------------------------#
# Evaluation Metrics                                                           #
# -----------------------------------------------------------------------------#


def evaluate(preds: Dict, trues: Dict, short_names: List[str]) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    metrics = {}

    for name in short_names:
        pred = preds[f"{name}"]  # median prediction
        true = trues[name]

        # Remove any NaN values
        mask = ~(np.isnan(pred) | np.isnan(true))
        pred = pred[mask]
        true = true[mask]

        if len(pred) == 0:
            continue

        # Calculate metrics
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))

        # Coverage: what fraction of true values fall within prediction intervals
        lower = preds[f"{name}_lower"][mask]
        upper = preds[f"{name}_upper"][mask]
        coverage = np.mean((true >= lower) & (true <= upper))

        metrics[f"{name}_mae"] = mae
        metrics[f"{name}_rmse"] = rmse
        metrics[f"{name}_coverage_90"] = coverage

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

    df_plot = pd.DataFrame(trues)
    for key, val in preds.items():
        df_plot[f"p_{key}"] = val

    for tk in sample_tickers:
        tid = _safe_ticker_id(ticker_map, tk)
        if tid is None:
            print(f"'{tk}' not found in ticker_map – skipping.")
            continue

        df_ticker = df_plot[df_plot["ticker"] == tid].sort_values("time_idx")

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
            plt.title(f"{tk} – {name.upper()} horizon")
            plt.ylabel("Log-return")
            plt.legend()

        plt.xlabel("Time index")
        plt.tight_layout()
        plt.savefig(out_dir / f"{tk}_timeseries.png")
        plt.close()
        print(f"Successfully generated plot for {tk}.")


def save_output(metrics: Dict, preds: Dict, trues: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w") as fp:
        json.dump({k: float(v) for k, v in metrics.items()}, fp, indent=2)

    pd.DataFrame({**trues, **preds}).to_csv(out_dir / "predictions.csv", index=False)
    print(f"\nResults written to {out_dir}")


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

    cp = torch.load(best, map_location="cpu", weights_only=False)
    ds_params = cp["hyper_parameters"]["timeseries_dataset_params"]

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

    full_dataset = TimeSeriesDataSet.from_parameters(ds_params, df_full, predict=False)
    print(f"Recreated full dataset with {len(full_dataset)} samples.")

    model = GlobalTFT.load_from_checkpoint(
        best, timeseries_dataset=full_dataset, map_location="cpu"
    )
    print("Model loaded successfully.")

    tickers_to_evaluate = _cfg_list(cfg.evaluate.sample_tickers)
    if not tickers_to_evaluate:
        sys.exit("No tickers specified in cfg.evaluate.sample_tickers. Aborting.")

    cfg_yaml = next(latest_run.glob("**/.hydra/config.yaml"), None)
    run_cfg = OmegaConf.load(cfg_yaml) if cfg_yaml else cfg
    group_id_col = _cfg_list(run_cfg.data.group_ids)[0]

    ticker_map_path = Path(cfg.paths.data_dir) / "processed" / "ticker_map.parquet"
    ticker_map = pl.read_parquet(ticker_map_path) if ticker_map_path.exists() else None

    if ticker_map is None:
        sys.exit("ticker_map.parquet not found. Cannot filter by ticker name.")

    filtered_map = ticker_map.filter(pl.col("ticker").is_in(tickers_to_evaluate))
    ticker_ids_to_evaluate_str = [
        str(i) for i in filtered_map.select("ticker_id").to_series().to_list()
    ]
    if not ticker_ids_to_evaluate_str:
        sys.exit(f"Could not find any of {tickers_to_evaluate} in the ticker_map.")

    eval_dataset = full_dataset.filter(
        lambda x: x[group_id_col].isin(ticker_ids_to_evaluate_str)
    )
    print(
        f"Filtered dataset to {len(eval_dataset)} samples for tickers: {tickers_to_evaluate}"
    )

    loader = eval_dataset.to_dataloader(
        train=False,
        batch_size=cfg.evaluate.batch_size,
        shuffle=False,
        num_workers=cfg.evaluate.num_workers,
    )
    print("Created dataloader for evaluation.")

    preds, trues, short_tgt_names = run_inference(model, loader, run_cfg, full_dataset)
    metrics = evaluate(preds, trues, short_tgt_names)

    print("\n--- Metrics for evaluated tickers ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    out_dir = Path(cfg.paths.log_dir) / "evaluation" / latest_run.name
    plot_preds(
        preds,
        trues,
        out_dir,
        ticker_map,
        tickers_to_evaluate,
        short_tgt_names,
    )
    save_output(metrics, preds, trues, out_dir)


if __name__ == "__main__":
    main()
