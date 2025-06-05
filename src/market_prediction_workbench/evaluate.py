# src/market_prediction_workbench/evaluate.py
import json
import sys  # Import sys to use sys.exit
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your custom modules
from market_prediction_workbench.model import GlobalTFT


# --- Helper functions from train.py (to avoid code duplication, consider moving to a utils.py file) ---
def get_list_from_cfg_node(config_node_val):
    if config_node_val is None:
        return []
    if isinstance(config_node_val, (str, int, float)):
        return [str(config_node_val)]
    # Check for list or ListConfig from OmegaConf
    if isinstance(config_node_val, (list, OmegaConf.get_type(OmegaConf.create([])))):
        return [str(item) for item in config_node_val]
    raise TypeError(
        f"Expected list or primitive for config node, got {type(config_node_val)}"
    )


def load_model_from_checkpoint(checkpoint_path: Path) -> Tuple[GlobalTFT, DictConfig]:
    """Load model from checkpoint and reconstruct its configuration."""
    if not checkpoint_path or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load to CPU to avoid potential GPU memory issues on a different machine
    model = GlobalTFT.load_from_checkpoint(
        str(checkpoint_path), map_location=torch.device("cpu")
    )
    model.eval()

    # The config is saved in hparams by our GlobalTFT wrapper
    if "timeseries_dataset_params" not in model.hparams:
        raise ValueError(
            "Cannot find 'timeseries_dataset_params' in model hyperparameters. Was the model saved correctly?"
        )

    # Reconstruct a simplified config from hparams for data preparation
    cfg = OmegaConf.create(
        {
            "data": model.hparams.timeseries_dataset_params,
            "evaluate": {"batch_size": 256, "num_workers": 4},  # Add defaults
        }
    )
    return model, cfg


def prepare_validation_dataloader(
    config: DictConfig, processed_data_path: Path
) -> Tuple[DataLoader, TimeSeriesDataSet]:
    """Prepares the validation dataloader using the same logic as in training."""
    data_pd = pl.read_parquet(processed_data_path).to_pandas()

    time_idx_col = str(config.data.time_idx)
    data_pd[time_idx_col] = data_pd[time_idx_col].astype(np.int64)

    static_cats = get_list_from_cfg_node(config.data.static_categoricals)
    for cat_col in static_cats:
        if cat_col in data_pd.columns:
            data_pd[cat_col] = data_pd[cat_col].astype(str)

    # Re-create the TimeSeriesDataSet exactly as it was during training
    dataset = TimeSeriesDataSet.from_parameters(
        dataset_params=config.data,
        data=data_pd,
        predict=False,  # We are not predicting, just creating loader for validation data
        stop_randomization=True,
    )

    # Create a dataloader for the validation set
    val_loader = dataset.to_dataloader(
        train=False,  # This is crucial for getting validation/test samples
        batch_size=config.evaluate.batch_size,
        shuffle=False,
        num_workers=config.evaluate.num_workers,
    )
    return val_loader, dataset


def run_inference(
    model: GlobalTFT, dataloader: DataLoader, config: DictConfig
) -> Tuple[Dict, Dict]:
    """Run model inference on a dataset and collect results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_actuals = [], []
    all_tickers, all_time_idx = [], []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Running inference"):
            x = {k: v.to(device) for k, v in x.items()}

            raw_output = model(x).prediction

            n_quantiles = len(model.model.loss.quantiles)
            n_targets = len(get_list_from_cfg_node(config.data.target))
            horizon = raw_output.shape[1]

            preds = raw_output.view(-1, horizon, n_targets, n_quantiles)

            all_preds.append(preds.cpu().numpy())
            all_actuals.append(y[0].cpu().numpy())

            all_tickers.append(x["groups"][:, 0].cpu().numpy())
            all_time_idx.append(x["decoder_time_idx"].cpu().numpy())

    predictions_np = np.concatenate(all_preds, axis=0)
    actuals_np = np.concatenate(all_actuals, axis=0)
    tickers_np = np.concatenate(all_tickers, axis=0)
    time_idx_np = np.concatenate(all_time_idx, axis=0)

    preds_h1 = predictions_np[:, 0, :, :]
    actuals_h1 = actuals_np[:, 0, :]
    time_idx_h1 = time_idx_np[:, 0]

    predictions_dict = {}
    actuals_dict = {"ticker": tickers_np, "time_idx": time_idx_h1}
    target_names = [
        t.replace("target_", "") for t in get_list_from_cfg_node(config.data.target)
    ]

    for i, name in enumerate(target_names):
        predictions_dict[f"{name}_lower"] = preds_h1[:, i, 0]
        predictions_dict[f"{name}"] = preds_h1[:, i, 1]
        predictions_dict[f"{name}_upper"] = preds_h1[:, i, 2]
        actuals_dict[f"{name}"] = actuals_h1[:, i]

    return predictions_dict, actuals_dict


def evaluate_performance(
    predictions: Dict, actuals: Dict, target_names: List[str]
) -> Dict[str, float]:
    """Calculate evaluation metrics for model performance."""
    metrics = {}
    for horizon in target_names:
        pred = predictions[horizon]
        true = actuals[horizon]
        lower = predictions[f"{horizon}_lower"]
        upper = predictions[f"{horizon}_upper"]

        metrics[f"{horizon}_mae"] = np.mean(np.abs(pred - true))
        metrics[f"{horizon}_coverage"] = np.mean((true >= lower) & (true <= upper))

        if np.sum(np.sign(true)) != 0:
            metrics[f"{horizon}_dir_accuracy"] = np.mean(np.sign(pred) == np.sign(true))

    return metrics


def plot_results(
    predictions: Dict,
    actuals: Dict,
    output_dir: Path,
    ticker_map: pl.DataFrame,
    sample_tickers: List[str],
    target_names: List[str],
):
    """Generate visualizations of model performance."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker_str in sample_tickers:
        try:
            ticker_id = ticker_map.filter(pl.col("ticker") == ticker_str)[
                "ticker_id"
            ].item()
        except (pl.exceptions.NoRowsReturnedError, IndexError):
            print(
                f"Warning: Ticker '{ticker_str}' not found in ticker_map. Skipping plot."
            )
            continue

        mask = actuals["ticker"] == ticker_id
        if not np.any(mask):
            continue

        plt.figure(figsize=(15, 4 * len(target_names)))
        time_indices = actuals["time_idx"][mask]
        sorted_idx = np.argsort(time_indices)

        for i, horizon in enumerate(target_names):
            plt.subplot(len(target_names), 1, i + 1)
            plt.plot(
                time_indices[sorted_idx],
                actuals[horizon][mask][sorted_idx],
                "b-",
                label="Actual",
            )
            plt.plot(
                time_indices[sorted_idx],
                predictions[horizon][mask][sorted_idx],
                "r--",
                label="Predicted Median",
            )
            plt.fill_between(
                time_indices[sorted_idx],
                predictions[f"{horizon}_lower"][mask][sorted_idx],
                predictions[f"{horizon}_upper"][mask][sorted_idx],
                alpha=0.2,
                color="orange",
                label="90% PI",
            )
            plt.title(f"{ticker_str} - {horizon.upper()} Horizon Prediction")
            plt.ylabel("Log Return")
            plt.legend()

        plt.xlabel("Time Index")
        plt.tight_layout()
        plt.savefig(output_dir / f"{ticker_str}_timeseries.png")
        plt.close()


def save_results(metrics: Dict, predictions: Dict, actuals: Dict, output_dir: Path):
    """Save evaluation results to files."""
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({k: round(v, 4) for k, v in metrics.items()}, f, indent=2)

    results_df = pd.DataFrame({**actuals, **predictions})
    results_df.to_csv(output_dir / "predictions.csv", index=False)
    print(f"\nResults saved to: {output_dir}")


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log_dir = Path(cfg.paths.log_dir)

    # --- NEW: Robustly find the best checkpoint from the latest run ---
    all_checkpoints = list(log_dir.glob("**/*.ckpt"))
    if not all_checkpoints:
        print(f"Error: No '.ckpt' files found under the log directory: {log_dir}")
        print(
            "Please run the training script (`train.py`) first to generate a model checkpoint."
        )
        sys.exit(1)

    run_dirs = {ckpt.parent for ckpt in all_checkpoints}
    latest_run_dir = max(
        run_dirs, key=lambda d: max(p.stat().st_mtime for p in d.glob("*.ckpt"))
    )
    print(f"Identified latest run directory: {latest_run_dir}")

    checkpoints_in_latest_run = list(latest_run_dir.glob("*.ckpt"))

    def get_val_loss(path):
        try:
            return float(path.stem.split("val_loss=")[-1].split("-")[0])
        except (ValueError, IndexError):
            return float("inf")

    checkpoints_in_latest_run.sort(key=get_val_loss)

    best_ckpt_with_loss = next(
        (
            ckpt
            for ckpt in checkpoints_in_latest_run
            if get_val_loss(ckpt) != float("inf")
        ),
        None,
    )

    if best_ckpt_with_loss:
        checkpoint_path = best_ckpt_with_loss
    else:
        print(
            f"Warning: Could not find a checkpoint with 'val_loss' in the name in {latest_run_dir}."
        )
        last_ckpt_path = latest_run_dir / "last.ckpt"
        if last_ckpt_path.exists():
            checkpoint_path = last_ckpt_path
            print("Using 'last.ckpt' as a fallback.")
        else:
            checkpoint_path = max(
                checkpoints_in_latest_run, key=lambda p: p.stat().st_mtime
            )
            print(
                f"Using the most recently modified checkpoint as a fallback: {checkpoint_path.name}"
            )
    # --- End of new checkpoint finding logic ---

    print(f"\nEvaluating best checkpoint from latest run: {checkpoint_path}")

    model, model_cfg = load_model_from_checkpoint(checkpoint_path)

    processed_data_file = Path(cfg.paths.processed_data_file)
    val_loader, dataset = prepare_validation_dataloader(model_cfg, processed_data_file)
    print(f"Prepared validation dataloader with {len(dataset)} samples.")

    predictions, actuals = run_inference(model, val_loader, model_cfg)

    target_names = [
        t.replace("target_", "") for t in get_list_from_cfg_node(model_cfg.data.target)
    ]
    metrics = evaluate_performance(predictions, actuals, target_names)
    print("\n--- Evaluation Metrics (on validation set) ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    run_output_dir = Path(cfg.paths.log_dir) / "evaluation" / latest_run_dir.name

    ticker_map_path = Path(cfg.paths.data_dir) / "processed" / "ticker_map.parquet"
    ticker_map = pl.read_parquet(ticker_map_path) if ticker_map_path.exists() else None

    if ticker_map is not None:
        plot_results(
            predictions,
            actuals,
            run_output_dir,
            ticker_map,
            cfg.evaluate.sample_tickers,
            target_names,
        )
    else:
        print("Warning: ticker_map.parquet not found. Skipping plots.")

    save_results(metrics, predictions, actuals, run_output_dir)


if __name__ == "__main__":
    main()
