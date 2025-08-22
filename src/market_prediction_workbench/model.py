# src/market_prediction_workbench/model.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from pytorch_forecasting.models.temporal_fusion_transformer._tft import (
    TemporalFusionTransformer,
)
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss
import numpy as np
from pytorch_forecasting.data import TimeSeriesDataSet
import pandas as pd
from omegaconf import ListConfig
import math

class TemporalFusionTransformerWithDevice(TemporalFusionTransformer):
    def get_attention_mask(self, *args, **kwargs) -> torch.Tensor:
        """
        Always create the attention mask on the model's parameter device.
        Causal mask for decoder, zero mask over encoder.
        """
        dev = next(self.parameters()).device  # <- force CUDA when model is on GPU

        # pull lengths
        enc = kwargs.get("encoder_lengths", kwargs.get("max_encoder_length"))
        dec = kwargs.get("decoder_lengths", kwargs.get("max_prediction_length"))

        def _as_len(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                if x.dim() == 0:
                    return int(x.item())
                return int(x.max().item())  # vector -> padded max
            return int(x)

        def _batch_from(x):
            return int(x.shape[0]) if (torch.is_tensor(x) and x.dim() >= 1) else None

        L_enc = _as_len(enc)
        L_dec = _as_len(dec)

        bs = kwargs.get("batch_size", None)
        if bs is None:
            bs = _batch_from(enc) or _batch_from(dec)
            if bs is None and len(args) >= 1:
                a0 = args[0]
                bs = int(a0.item()) if torch.is_tensor(a0) else int(a0)
        if bs is None:
            bs = 1

        # fallback to positional (PF sometimes passes (bs, L_enc, L_dec))
        if (L_enc is None or L_dec is None) and len(args) == 3:
            _, enc_pos, dec_pos = args
            L_enc = L_enc or _as_len(enc_pos)
            L_dec = L_dec or _as_len(dec_pos)

        if L_enc is None or L_dec is None:
            raise RuntimeError(f"Could not infer encoder/decoder lengths (args={args}, kwargs={kwargs}).")

        dec_mask = torch.triu(torch.ones((L_dec, L_dec), dtype=torch.bool, device=dev), diagonal=1)
        dec_mask = dec_mask.unsqueeze(0).expand(bs, L_dec, L_dec)
        enc_mask = torch.zeros((L_dec, L_enc), dtype=torch.bool, device=dev)
        enc_mask = enc_mask.unsqueeze(0).expand(bs, L_dec, L_enc)

        mask = torch.cat([dec_mask, enc_mask], dim=-1)
        # belt & suspenders:
        if mask.device != dev:
            mask = mask.to(dev)
        return mask


class GlobalTFT(pl.LightningModule):
    def __init__(
        self,
        model_specific_params: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        timeseries_dataset: TimeSeriesDataSet | None = None,
        timeseries_dataset_params: dict | None = None,
        lr_schedule: dict | None = None,
        steps_per_epoch: int | None = None,
        max_epochs: int | None = None,
    ):
        super().__init__()

        # Require a real dataset (training passes train_dataset; evaluation passes full_dataset)
        if timeseries_dataset is None and timeseries_dataset_params is None:
            raise ValueError(
                "Either `timeseries_dataset` or `timeseries_dataset_params` must be provided."
            )

        # Normalize accidental dict usage
        if isinstance(timeseries_dataset, dict) and timeseries_dataset_params is None:
            timeseries_dataset_params = timeseries_dataset
            timeseries_dataset = None

        # Build a dataset object we can rely on (`ds`) and collect its parameters
        if isinstance(timeseries_dataset, TimeSeriesDataSet):
            ds = timeseries_dataset
            ts_params = ds.get_parameters()
        else:
            ds = TimeSeriesDataSet.from_parameters(
                timeseries_dataset_params, pd.DataFrame(), predict=True
            )
            ts_params = timeseries_dataset_params

        # ---------------- loss & output_size handling ----------------
        # Do NOT overwrite output_size if it exists in model_specific_params (to match checkpoints).
        loss_instance = model_specific_params.get("loss", None)
        if loss_instance is None:
            loss_instance = QuantileLoss()  # safe default if not given

        # Normalize quantiles (Hydra may give ListConfig, strings, np arrays, etc.)
        if hasattr(loss_instance, "quantiles"):
            q = loss_instance.quantiles
            if isinstance(q, ListConfig):
                q = list(q)
            elif isinstance(q, np.ndarray):
                q = q.tolist()
            elif isinstance(q, (float, int)):
                q = [float(q)]
            elif isinstance(q, (list, tuple)):
                q = list(q)
            else:
                # last resort: wrap as single-element list
                q = [float(q)]
            # ensure float list
            q = [float(x) for x in q]
            # write back a plain python list to the loss so downstream is clean
            loss_instance.quantiles = q
            qn = len(q)
        else:
            qn = 1  # non-quantile losses

        # If no output_size provided, infer from the number of quantiles and number of targets.
        if "output_size" not in model_specific_params:
            n_targets = len(ds.target_names)
            model_specific_params["output_size"] = qn if n_targets == 1 else [qn] * n_targets

        # Clean params so nn.Modules are not stored in hparams
        init_model_params_copy = model_specific_params.copy()
        cleaned_model_params = {
            k: v
            for k, v in init_model_params_copy.items()
            if not isinstance(v, (nn.Module, torch.nn.modules.loss._Loss))
        }

        # Save hyperparameters for checkpointing (store dataset parameters, not the object)
        self.save_hyperparameters(
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "model_specific_params": cleaned_model_params,
                "timeseries_dataset_params": ts_params,
                "lr_schedule": lr_schedule or {},
                "steps_per_epoch": int(steps_per_epoch) if steps_per_epoch is not None else None,
                "max_epochs": int(max_epochs) if max_epochs is not None else None,
            }
        )

        # Build TFT from the actual dataset. Pass the loss explicitly and the cleaned params.
        self.model = TemporalFusionTransformerWithDevice.from_dataset(
            ds,
            loss=loss_instance,
            learning_rate=learning_rate,
            **cleaned_model_params,
        )

        # Ensure loss modules know the prediction length (helps calibration-aware losses)
        dataset_max_pred_len = ds.max_prediction_length
        current_loss_module = self.model.loss
        if isinstance(current_loss_module, MultiLoss):
            for metric in current_loss_module.metrics:
                if hasattr(metric, "max_prediction_length") and metric.max_prediction_length is None:
                    metric.max_prediction_length = dataset_max_pred_len
                    print(
                        f"DEBUG GlobalTFT.__init__: Set metric {type(metric).__name__}.max_prediction_length "
                        f"to {dataset_max_pred_len}"
                    )
        elif hasattr(current_loss_module, "max_prediction_length"):
            if current_loss_module.max_prediction_length is None:
                current_loss_module.max_prediction_length = dataset_max_pred_len
                print(
                    f"DEBUG GlobalTFT.__init__: Set self.model.loss {type(current_loss_module).__name__}"
                    f".max_prediction_length to {dataset_max_pred_len}"
                )
            print(
                f"DEBUG GlobalTFT.__init__: self.model.loss ({type(current_loss_module).__name__})."
                f"max_prediction_length = {current_loss_module.max_prediction_length}"
            )
        else:
            print(
                f"DEBUG GlobalTFT.__init__: self.model.loss ({type(current_loss_module).__name__}) "
                f"does not have max_prediction_length attribute."
            )

        # Keep PF logging quiet/lean
        self.model.logging_metrics = nn.ModuleList([])
        print("DEBUG: model.logging_metrics has been cleared for debugging.")

    # ---- attach trainer & proxy logs to inner PF model ----
    def on_fit_start(self):
        self.model.trainer = self.trainer
        self.model.log = lambda *a, **k: self.log(*a, **k)
        self.model.log_dict = lambda *a, **k: self.log_dict(*a, **k)

    # ---------------- small helpers for robust tensor casting ----------------
    def _process_input_data(self, data_element):
        if torch.is_tensor(data_element):
            return data_element.to(device=self.device, dtype=torch.float32)
        elif isinstance(data_element, np.ndarray):
            if data_element.dtype == object:
                if (
                    data_element.ndim == 0
                    and hasattr(data_element.item(), "to")
                    and hasattr(data_element.item(), "device")
                ):
                    return data_element.item().to(device=self.device, dtype=torch.float32)
                processed_list = [self._process_input_data(el) for el in data_element]
                return torch.stack(processed_list) if processed_list else torch.empty(0, dtype=torch.float32, device=self.device)
            else:
                return torch.from_numpy(data_element).to(device=self.device, dtype=torch.float32)
        elif isinstance(data_element, (list, tuple)):
            if not data_element:
                return torch.empty(0, dtype=torch.float32, device=self.device)
            processed = [self._process_input_data(item) for item in data_element]
            try:
                return torch.stack(processed)
            except RuntimeError:
                return torch.tensor(data_element, dtype=torch.float32, device=self.device)
        else:
            return torch.tensor(data_element, dtype=torch.float32, device=self.device)

    def _prepare_target_tensor(self, raw_target_data):
        target = self._process_input_data(raw_target_data)
        if isinstance(target, list):
            return [t.unsqueeze(1) if t.ndim == 1 else t for t in target]
        if target.ndim == 2:
            target = target.unsqueeze(1)  # Add time dimension
        return target

    def _prepare_scale_tensor(self, raw_scale_data, target_shape):
        scale = self._process_input_data(raw_scale_data)
        if scale.ndim == 1 and scale.shape[0] == target_shape[0]:
            scale = scale.unsqueeze(-1)
        return scale

    # ---------------- Lightning plumbing ----------------
    def forward(self, x_batch):
        return self.model(x_batch)

    def training_step(self, batch, batch_idx):
        self.model.trainer = self.trainer
        return self.model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.model.trainer = self.trainer
        return self.model.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            fused=False,  # safer on memory across driver versions
        )
        sch_cfg = self.hparams.get("lr_schedule", {}) or {}
        sch_type = str(sch_cfg.get("type", "one_cycle")).lower()
        warmup_frac = float(sch_cfg.get("warmup_frac", 0.1))
        total_steps = int(self.hparams.get("steps_per_epoch", 100)) * int(self.hparams.get("max_epochs", 10))
        warmup_steps = max(1, int(total_steps * warmup_frac))

        if sch_type == "none":
            return {"optimizer": optimizer}

        from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LambdaLR, SequentialLR

        steps_per_epoch = self.hparams.get("steps_per_epoch")
        max_epochs = self.hparams.get("max_epochs")

        if steps_per_epoch and max_epochs:
            total_steps = int(steps_per_epoch) * int(max_epochs)
        else:
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)

        if not total_steps or total_steps <= 1:
            return {"optimizer": optimizer}

        if sch_type == "one_cycle":
            sched = OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=total_steps,
                pct_start=max(1e-4, min(0.5, warmup_frac)),
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=1e3,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        if sch_type == "cosine_warmup":
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step + 1) / warmup_steps
                # cosine decay from 1 -> 0 (eta_min handled by optimizer if you prefer)
                progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
                "monitor": None,
            }
        return {"optimizer": optimizer}
