# src/market_prediction_workbench/model.py
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from torch.optim import AdamW
import torch.nn as nn  # For type hint


class GlobalTFT(pl.LightningModule):
    def __init__(
        self,
        timeseries_dataset: TimeSeriesDataSet,
        model_specific_params: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()

        # Make a copy of model_specific_params to not modify the dictionary that might be used elsewhere.
        init_model_specific_params_copy = model_specific_params.copy()

        # Prepare hyperparameters for GlobalTFT to save.
        # We'll store a version of model_specific_params that excludes nn.Module instances.
        model_specific_params_for_hparams = {
            k: v
            for k, v in init_model_specific_params_copy.items()
            if not isinstance(v, nn.Module)
        }

        # Save hyperparameters for the GlobalTFT module.
        # 'model_specific_params' saved here will be the cleaned version.
        self.save_hyperparameters(
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "model_specific_params": model_specific_params_for_hparams,
            }
        )
        # Note: self.hparams will now contain these.
        # self.hparams.model_specific_params will be the cleaned dict.

        # The actual TemporalFusionTransformer instance will receive the original
        # model_specific_params, which may include nn.Module instances like 'loss'.
        # TFT's own save_hyperparameters mechanism handles ignoring these internally.
        self.model = TemporalFusionTransformer.from_dataset(
            timeseries_dataset,
            **init_model_specific_params_copy,  # Pass the original params, including any nn.Modules
        )

    def forward(self, x_batch):  # x_batch is the input dict from dataloader
        output = self.model(x_batch)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        # self.model.loss refers to the loss function instance held by the TFT model
        loss = self.model.loss(
            out, y[0], y[1] if len(y) > 1 else None
        )  # y[1] is target_scale/weights
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.model.loss(out, y[0], y[1] if len(y) > 1 else None)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,  # Correctly accesses saved hyperparameter
            weight_decay=self.hparams.weight_decay,  # Correctly accesses saved hyperparameter
        )
        return optimizer
