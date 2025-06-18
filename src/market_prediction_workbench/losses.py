# src/market_prediction_workbench/losses.py

import torch
import torch.nn as nn
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric


class DirectionalLoss(MultiHorizonMetric):
    """
    A loss function that penalizes incorrect directional predictions more heavily.

    This loss combines Mean Squared Error (MSE) with a directional penalty.
    If the sign of the prediction is different from the sign of the target,
    an additional penalty is applied.

    Args:
        directional_penalty (float): The factor by which to multiply the MSE
                                     when the predicted direction is wrong.
                                     A value > 1.0. Defaults to 2.0.
    """

    def __init__(self, directional_penalty: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        if directional_penalty <= 1.0:
            raise ValueError("Directional penalty must be greater than 1.0")
        self.directional_penalty = directional_penalty
        # We want per-element losses to apply weights and let the parent class handle reduction
        self.mse_loss = nn.MSELoss(reduction="none")

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for a single target. The MultiHorizonMetric base class
        will handle iterating over multiple targets.

        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: A tensor of loss values, one for each sample in the batch.
                          The shape should be (batch_size, prediction_horizon).
        """
        # TFT outputs quantiles. For a point-forecast loss, we should use the median prediction.
        # The median is typically the center of the last dimension.
        if y_pred.ndim == y_true.ndim + 1 and y_pred.shape[-1] > 1:
            median_pred_index = y_pred.shape[-1] // 2
            y_pred = y_pred[..., median_pred_index]

        # --- FIX FOR USERWARNING ---
        # Explicitly align shapes to prevent broadcasting warnings from PyTorch.
        # The MultiHorizonMetric can sometimes pass tensors with mismatched shapes.
        if y_pred.shape != y_true.shape:
            y_true = y_true.expand_as(y_pred)
        # --- END FIX ---

        # Calculate standard MSE loss
        mse = self.mse_loss(y_pred, y_true)

        # Calculate directional penalty weights
        # torch.sign returns -1, 0, 1. Multiplying signs will be > 0 if they are the same.
        signs_match = torch.sign(y_pred) * torch.sign(y_true)

        # Apply penalty where signs do not match (signs_match <= 0)
        weights = torch.where(
            signs_match <= 0,
            self.directional_penalty,
            1.0,
        )

        # Apply weights to the MSE loss
        weighted_mse = mse * weights

        # Return the tensor of losses for each item in the batch.
        # DO NOT call .mean() here. The parent class handles the reduction.
        return weighted_mse
