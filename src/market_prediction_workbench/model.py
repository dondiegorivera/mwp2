# src/market_prediction_workbench/model.py
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from torch.optim import AdamW

# import torch.nn.functional as F # If doing manual MAE logging etc.


class GlobalTFT(pl.LightningModule):
    def __init__(
        self,
        timeseries_dataset: TimeSeriesDataSet,  # Pass the whole dataset
        model_specific_params: dict,  # Params like hidden_size, lstm_layers, dropout etc.
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()

        # Save hyperparameters.
        # We don't save timeseries_dataset directly as it can be large and complex.
        # Its relevant parameters are used by from_dataset.
        self.save_hyperparameters(
            "model_specific_params", "learning_rate", "weight_decay"
        )

        # The model_specific_params should contain:
        # hidden_size, lstm_layers, dropout, attention_head_size, etc.
        # It will also determine the output_size and loss if not defaults.
        # For example, if model_specific_params includes "quantiles": [0.05, 0.5, 0.95],
        # from_dataset will configure QuantileLoss and output_size accordingly.

        self.model = TemporalFusionTransformer.from_dataset(
            timeseries_dataset,  # Use the passed dataset
            **self.hparams.model_specific_params,  # Unpack model-specific parameters
        )

        # The loss function is typically handled by the model if created from_dataset with quantiles
        # self.loss_fn = self.model.loss # This would get the loss function from the TFT instance

    # forward, training_step, validation_step, configure_optimizers remain the same for now.
    # ... (rest of the class as before) ...

    def forward(self, x_batch):  # x_batch is the input dict from dataloader
        output = self.model(x_batch)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
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

        # Example: Log MAE for the median forecast (P50) for the first target variable
        # This assumes "target_1d" is the first target and we have a median prediction
        # if 0.5 in self.model.hparams.loss.quantiles:
        #     median_idx = self.model.hparams.loss.quantiles.index(0.5)
        #     # y[0] is (batch_size, decoder_length, num_targets)
        #     # out["prediction"] is (batch_size, decoder_length, num_quantiles_x_num_targets)
        #     # If multi-target, predictions are concatenated along the last dim.
        #     # Assuming single target for simplicity here, or adjust indexing
        #     p50_predictions = out["prediction"][..., median_idx]
        #     actuals = y[0][..., 0] # Assuming first target

        #     # Handle target scaling if TimeSeriesDataSet used it
        #     if y[1] is not None and "target_scale" in x: # y[1] might be weights or target_scale
        #          # This depends on how TimeSeriesDataSet structures y and if target_scale is directly available
        #          # Often target_scale is part of x (decoder_target_scale or encoder_target_scale)
        #          # For simplicity, assuming direct comparison or manual unscaling needed elsewhere.
        #          pass

        #     mae = F.l1_loss(p50_predictions, actuals)
        #     self.log("val_mae_p50_target0", mae, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
