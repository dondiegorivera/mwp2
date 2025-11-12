import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from market_prediction_workbench.model import GlobalTFT
from market_prediction_workbench.sampling import (
    sequence_balance_weights,
    tail_upweighting,
)
from market_prediction_workbench.train import RankICCallback


def test_sequence_balance_weights():
    seq_ids = pd.Series(["A", "A", "B", "C", "C", "C"])
    weights = sequence_balance_weights(seq_ids)
    # sequences with more samples should receive lower weights
    assert weights[0] == weights[1]  # same seq
    assert weights[2] > weights[0]  # seq B (1 sample) > seq A (2 samples)
    assert np.isclose(weights[3], weights[4])
    assert np.isclose(weights[3], 1 / 3)


def test_tail_upweighting_percentiles():
    arr = np.array([0.1, 0.2, 0.5, 1.0, 5.0, 10.0])
    weights = tail_upweighting(arr, q_low=50, q_high=80, mid_scale=1.5, high_scale=3.0)
    # ensure baseline ones for small values
    assert weights[0] == 1.0
    # check that extreme tail gets high_scale
    assert weights[-1] == 3.0
    # NaNs should map to 1.0
    arr_with_nan = np.array([np.nan, 0.5, 2.0])
    weights_nan = tail_upweighting(arr_with_nan)
    assert weights_nan[0] == 1.0


def _build_dummy_dataset():
    data = pd.DataFrame(
        {
            "time_idx": np.arange(30),
            "ticker_id": ["A"] * 15 + ["B"] * 15,
            "target_5d": np.linspace(-0.1, 0.1, 30),
        }
    )
    return TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="target_5d",
        group_ids=["ticker_id"],
        max_encoder_length=5,
        max_prediction_length=1,
        add_relative_time_idx=True,
        add_target_scales=True,
    )


def _build_model(lr_schedule: dict | None):
    dataset = _build_dummy_dataset()
    model_params = {
        "hidden_size": 8,
        "lstm_layers": 1,
        "dropout": 0.1,
        "output_size": 1,
    }
    return GlobalTFT(
        timeseries_dataset=dataset,
        model_specific_params=model_params,
        learning_rate=1e-3,
        lr_schedule=lr_schedule,
        steps_per_epoch=2,
        max_epochs=2,
    )


def test_configure_optimizers_one_cycle():
    module = _build_model({"type": "one_cycle", "warmup_frac": 0.1})
    opt_dict = module.configure_optimizers()
    scheduler = opt_dict["lr_scheduler"]["scheduler"]
    from torch.optim.lr_scheduler import OneCycleLR

    assert isinstance(scheduler, OneCycleLR)


def test_configure_optimizers_cosine_warmup():
    module = _build_model({"type": "cosine_warmup", "warmup_frac": 0.2})
    opt_dict = module.configure_optimizers()
    scheduler = opt_dict["lr_scheduler"]["scheduler"]
    from torch.optim.lr_scheduler import LambdaLR

    assert isinstance(scheduler, LambdaLR)


def test_rank_ic_callback_spearman():
    class DummyDataset:
        target_normalizer = None

    callback = RankICCallback(dataset=DummyDataset(), target_idx=0, horizon=1)
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0.4, 0.3, 0.2, 0.1])
    score = callback._spearman(x, y)
    assert np.isclose(score, -1.0)
