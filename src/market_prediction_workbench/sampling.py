"""Helpers for sampler weight calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sequence_balance_weights(sequence_ids: pd.Series | np.ndarray) -> np.ndarray:
    """
    Return weights proportional to 1 / count(sequence_id).
    """
    series = (
        sequence_ids if isinstance(sequence_ids, pd.Series) else pd.Series(sequence_ids)
    )
    counts = series.value_counts()
    weights = series.map(1.0 / counts).astype("float64").to_numpy()
    return weights


def tail_upweighting(
    abs_targets: np.ndarray,
    *,
    q_low: float = 70.0,
    q_high: float = 90.0,
    mid_scale: float = 1.5,
    high_scale: float = 2.0,
) -> np.ndarray:
    """
    Compute multiplicative tail weights based on absolute target magnitude.
    """
    abs_targets = np.asarray(abs_targets, dtype=float)
    finite = np.isfinite(abs_targets)
    weights = np.ones_like(abs_targets, dtype=float)
    if not finite.any():
        return weights

    ql, qh = np.nanpercentile(abs_targets[finite], [q_low, q_high])
    weights[(abs_targets >= ql) & (abs_targets < qh)] = mid_scale
    weights[abs_targets >= qh] = high_scale
    weights[~finite] = 1.0
    return weights
