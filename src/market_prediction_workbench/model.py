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

# Added for the new __init__ logic
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd


class TemporalFusionTransformerWithDevice(TemporalFusionTransformer):
    def get_attention_mask(self, *args, **kwargs) -> torch.Tensor:
        if len(args) == 3 and all(isinstance(x, (int, torch.Tensor)) for x in args):
            raw_bs, raw_enc, raw_dec = args

            def to_int(x):
                return int(x.item()) if torch.is_tensor(x) else int(x)

            batch_size, L_enc, L_dec = to_int(raw_bs), to_int(raw_enc), to_int(raw_dec)
            mask_device = self.device
        else:
            raw_enc = kwargs.get("encoder_lengths", kwargs.get("max_encoder_length"))
            raw_dec = kwargs.get("decoder_lengths", kwargs.get("max_prediction_length"))
            if raw_enc is None or raw_dec is None:
                raise RuntimeError(
                    f"Could not parse lengths from args={args}, kwargs={kwargs}"
                )
            if torch.is_tensor(raw_enc) and raw_enc.dim() >= 1:
                batch_size = raw_enc.shape[0]
            elif torch.is_tensor(raw_dec) and raw_dec.dim() >= 1:
                batch_size = raw_dec.shape[0]
            else:
                raise RuntimeError(
                    f"Cannot infer batch_size (got {raw_enc}, {raw_dec})"
                )

            def to_int(x):
                return (
                    int(x[0].item())
                    if torch.is_tensor(x) and x.numel() > 1
                    else (int(x.item()) if torch.is_tensor(x) else int(x))
                )

            L_enc, L_dec = to_int(raw_enc), to_int(raw_dec)
            if torch.is_tensor(raw_enc):
                mask_device = raw_enc.device
            elif torch.is_tensor(raw_dec):
                mask_device = raw_dec.device
            else:
                mask_device = self.device

        decoder_mask = torch.triu(
            torch.ones((L_dec, L_dec), dtype=torch.bool, device=mask_device), diagonal=1
        )
        decoder_mask = decoder_mask.unsqueeze(0).expand(batch_size, L_dec, L_dec)
        encoder_mask = torch.zeros((L_dec, L_enc), dtype=torch.bool, device=mask_device)
        encoder_mask = encoder_mask.unsqueeze(0).expand(batch_size, L_dec, L_enc)
        return torch.cat([decoder_mask, encoder_mask], dim=-1)


class GlobalTFT(pl.LightningModule):
    def __init__(
        self,
        model_specific_params: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        timeseries_dataset: TimeSeriesDataSet | None = None,
        timeseries_dataset_params: dict | None = None,
    ):
        super().__init__()

        if timeseries_dataset is None and timeseries_dataset_params is None:
            raise ValueError(
                "Either `timeseries_dataset` or `timeseries_dataset_params` must be provided."
            )

        # If loading from checkpoint, timeseries_dataset might be None.
        # Reconstruct a skeleton dataset from params to initialize the model.
        if timeseries_dataset is None:
            # This allows loading a model without having to load the data first.
            # The actual data is needed for the dataloaders, but not for model architecture.
            timeseries_dataset = TimeSeriesDataSet.from_parameters(
                timeseries_dataset_params, pd.DataFrame(), predict=True
            )

        # The original __init__ logic can proceed from here.
        num_targets = len(timeseries_dataset.target_names)

        loss_instance = model_specific_params.get("loss")
        quantiles = (
            loss_instance.quantiles if hasattr(loss_instance, "quantiles") else [0.5]
        )
        output_size = len(quantiles)
        if "output_size" not in model_specific_params:
            model_specific_params["output_size"] = output_size

        # We save the parameters, not the full dataset object.
        # The model_specific_params might contain non-serializable objects (like the loss function instance)
        # so we clean it for saving.

        if isinstance(model_specific_params["output_size"], int):
            model_specific_params["output_size"] = [
                model_specific_params["output_size"]
            ] * num_targets

        init_model_params_copy = model_specific_params.copy()

        cleaned_model_params = {
            k: v
            for k, v in init_model_params_copy.items()
            if not isinstance(v, (nn.Module, torch.nn.modules.loss._Loss))
        }

        # Save hyperparameters for checkpointing. This is crucial for reloading.
        self.save_hyperparameters(
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "model_specific_params": cleaned_model_params,
                "timeseries_dataset_params": timeseries_dataset.get_parameters(),
            }
        )

        self.model = TemporalFusionTransformerWithDevice.from_dataset(
            timeseries_dataset,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **init_model_params_copy,  # Pass original params with loss object
        )

        dataset_max_pred_len = timeseries_dataset.max_prediction_length
        current_loss_module = self.model.loss
        if isinstance(current_loss_module, MultiLoss):
            for metric in current_loss_module.metrics:
                if (
                    hasattr(metric, "max_prediction_length")
                    and metric.max_prediction_length is None
                ):
                    metric.max_prediction_length = dataset_max_pred_len
                    print(
                        f"DEBUG GlobalTFT.__init__: Set metric {type(metric).__name__}.max_prediction_length to {dataset_max_pred_len}"
                    )
        elif hasattr(current_loss_module, "max_prediction_length"):
            if current_loss_module.max_prediction_length is None:
                current_loss_module.max_prediction_length = dataset_max_pred_len
                print(
                    f"DEBUG GlobalTFT.__init__: Set self.model.loss {type(current_loss_module).__name__}.max_prediction_length to {dataset_max_pred_len}"
                )
            # Debug print for confirmation
            print(
                f"DEBUG GlobalTFT.__init__: self.model.loss ({type(current_loss_module).__name__}).max_prediction_length = {current_loss_module.max_prediction_length}"
            )

        else:
            print(
                f"DEBUG GlobalTFT.__init__: self.model.loss ({type(current_loss_module).__name__}) does not have max_prediction_length attribute."
            )

        # Temporary override to simplify debugging:
        self.model.logging_metrics = nn.ModuleList([])
        print("DEBUG: model.logging_metrics has been cleared for debugging.")

    # ... _process_input_data, _prepare_target_tensor, _prepare_scale_tensor ...
    def _process_input_data(self, data_element):
        # print(f"DEBUG _process_input_data: received type {type(data_element)}")
        if torch.is_tensor(data_element):
            # print(f"DEBUG _process_input_data: is_tensor, shape {data_element.shape}")
            return data_element.to(device=self.device, dtype=torch.float32)
        elif isinstance(data_element, np.ndarray):
            # print(f"DEBUG _process_input_data: is_ndarray, dtype {data_element.dtype}, shape {data_element.shape}")
            if data_element.dtype == object:
                if (
                    data_element.ndim == 0
                    and hasattr(data_element.item(), "to")
                    and hasattr(data_element.item(), "device")
                ):
                    return data_element.item().to(
                        device=self.device, dtype=torch.float32
                    )
                try:
                    processed_list = [
                        self._process_input_data(el) for el in data_element
                    ]
                    return (
                        torch.stack(processed_list)
                        if processed_list
                        else torch.empty(0, dtype=torch.float32, device=self.device)
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to process numpy array of dtype 'object'. Shape: {data_element.shape}. "
                        f"First element type: {type(data_element.flat[0]) if data_element.size > 0 else 'N/A'}. Error: {e}"
                    ) from e
            else:
                return torch.from_numpy(data_element).to(
                    device=self.device, dtype=torch.float32
                )
        elif isinstance(data_element, (list, tuple)):
            # print(f"DEBUG _process_input_data: is_list_or_tuple, len {len(data_element)}")
            if not data_element:
                return torch.empty(0, dtype=torch.float32, device=self.device)
            processed_elements = [
                self._process_input_data(item) for item in data_element
            ]
            try:
                return torch.stack(processed_elements)
            except RuntimeError as e:
                try:
                    return torch.tensor(
                        data_element, dtype=torch.float32, device=self.device
                    )
                except Exception as e_tensor:
                    shapes = [
                        (
                            f"{type(pe)}:{pe.shape}"
                            if torch.is_tensor(pe)
                            else str(type(pe))
                        )
                        for pe in processed_elements
                    ]
                    raise RuntimeError(
                        f"Failed to stack OR create tensor from processed elements from list/tuple. Original type: {type(data_element)}. "
                        f"Processed element types/shapes: {shapes}. Stack error: {e}. Tensor creation error: {e_tensor}"
                    ) from e_tensor
        else:
            try:
                # print(f"DEBUG _process_input_data: is_other_scalar, value {data_element}")
                return torch.tensor(
                    data_element, dtype=torch.float32, device=self.device
                )
            except Exception as e:
                raise TypeError(
                    f"Unsupported type {type(data_element)} for _process_input_data. Error: {e}"
                ) from e

    def _prepare_target_tensor(self, raw_target_data):
        target = self._process_input_data(raw_target_data)

        # This was part of the original file and needs to be adapted for lists
        if isinstance(target, list):
            return [t.unsqueeze(1) if t.ndim == 1 else t for t in target]

        if target.ndim == 2:
            target = target.unsqueeze(1)  # Add time dimension
        return target

    def _prepare_scale_tensor(self, raw_scale_data, target_shape):
        scale = self._process_input_data(raw_scale_data)
        if (
            scale.ndim == 1 and scale.shape[0] > 0 and scale.shape[0] == target_shape[0]
        ):  # (B,)
            scale = scale.unsqueeze(-1)  # (B,1)
        return scale

    # ... forward, training_step, validation_step, configure_optimizers (no changes here) ...
    def forward(self, x_batch):
        return self.model(x_batch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        target, _ = y  # y is a tuple of (target, scale)

        # --- THE CORRECT FIX for both AttributeError and IndexError ---
        # The loss function expects a time dimension for each target.
        # Since max_prediction_horizon=1, targets from the dataloader are 1D.
        # We must add a time dimension of size 1.
        if isinstance(target, list):
            # If multi-target, reshape each tensor in the list.
            # Shape changes from [(B,), (B,), ...] to [(B, 1), (B, 1), ...]
            reshaped_target = [t.unsqueeze(1) for t in target]
        else:
            # If single-target, reshape the single tensor.
            # Shape changes from (B,) to (B, 1)
            reshaped_target = target.unsqueeze(1)
        # --- END FIX ---

        loss_val = self.model.loss(out.prediction, reshaped_target)
        bs = (
            reshaped_target[0].size(0)
            if isinstance(reshaped_target, list)
            else reshaped_target.size(0)
        )
        self.log(
            "train_loss",
            loss_val,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )
        return loss_val

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        target, _ = y

        # --- THE CORRECT FIX (Applied to validation as well) ---
        if isinstance(target, list):
            reshaped_target = [t.unsqueeze(1) for t in target]
        else:
            reshaped_target = target.unsqueeze(1)
        # --- END FIX ---

        loss_val = self.model.loss(out.prediction, reshaped_target)
        self.log("val_loss", loss_val)
        return loss_val

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            fused=True if torch.cuda.is_available() else False,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            factor=0.5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
