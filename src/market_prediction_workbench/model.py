# src/market_prediction_workbench/model.py

import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from torch.optim import AdamW
import torch.nn as nn


class GlobalTFT(pl.LightningModule):
    def __init__(
        self,
        timeseries_dataset: TimeSeriesDataSet,
        model_specific_params: dict,  # This will contain 'loss', possibly 'embedding_sizes', etc.
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()

        # Make a copy to avoid modifying the original dict if it's reused elsewhere
        init_model_params_copy = model_specific_params.copy()

        # Compute embedding sizes only after the dataset’s encoders exist
        # This ensures keys like "ticker_id" are present
        embedding_sizes = timeseries_dataset.get_embedding_sizes()
        if embedding_sizes:
            init_model_params_copy["embedding_sizes"] = embedding_sizes
        else:
            # If for some reason the dict contained an outdated or empty embedding_sizes, remove it
            init_model_params_copy.pop("embedding_sizes", None)

        # Prepare hyperparameters to save (exclude the large dataset and any nn.Modules)
        hparams_for_global_tft = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "model_specific_params": {
                k: v
                for k, v in init_model_params_copy.items()
                if not isinstance(v, nn.Module)
            },
        }
        # Ignore the dataset itself when saving hyperparameters
        self.save_hyperparameters(hparams_for_global_tft, ignore=["timeseries_dataset"])

        # Instantiate the TemporalFusionTransformer using the corrected params dict
        self.model = TemporalFusionTransformer.from_dataset(
            timeseries_dataset,
            learning_rate=learning_rate,  # TFT’s internal optimizer LR
            weight_decay=weight_decay,  # TFT’s internal optimizer weight decay
            **init_model_params_copy,  # Contains 'loss' (nn.Module) and now correct 'embedding_sizes'
        )

    def forward(self, x_batch):
        """
        x_batch is the dictionary of input tensors from TimeSeriesDataSet.to_dataloader().
        The TFT model expects that dict and returns a model-specific output object.
        """
        return self.model(x_batch)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: input dict, y: (target_tensor, optional_weight_tensor)
        out = self(x)  # calls forward -> TFT output object

        # Compute loss; TFT.loss usually expects (prediction, target, optional weight)
        if len(y) > 1 and y[1] is not None:
            loss = self.model.loss(out.prediction, y[0], y[1])
        else:
            loss = self.model.loss(out.prediction, y[0])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        if len(y) > 1 and y[1] is not None:
            loss = self.model.loss(out.prediction, y[0], y[1])
        else:
            loss = self.model.loss(out.prediction, y[0])

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
