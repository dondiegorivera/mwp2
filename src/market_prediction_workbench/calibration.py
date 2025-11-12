"""Shared helpers for interval coverage and calibration."""

from __future__ import annotations

import numpy as np

Z_90 = 1.645  # z-score for central 90% interval


def _as_array(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    return np.asarray(x, dtype=float)


def coverage_fraction(truth, lower, upper, *, default: float = float("nan")) -> float:
    """
    Compute fraction of truth values inside [lower, upper].

    Returns NaN (or ``default``) if no finite comparisons exist.
    """
    y = _as_array(truth)
    lo = _as_array(lower)
    hi = _as_array(upper)
    if y.size == 0 or lo.size == 0 or hi.size == 0:
        return default
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(mask):
        return default
    return float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])))


def estimate_calibration_alpha(
    truth, median, lower, upper, percentile: float = 90.0
) -> float:
    """
    Estimate multiplicative alpha for predictive intervals so that
    |residual| / (Z_90 * s_hat) roughly matches the requested percentile.
    """
    y = _as_array(truth)
    mid = _as_array(median)
    lo = _as_array(lower)
    hi = _as_array(upper)
    if any(arr.size == 0 for arr in (y, mid, lo, hi)):
        return float("nan")

    s_hat = (hi - lo) / (2.0 * Z_90)
    denom = Z_90 * s_hat
    resid = y - mid
    mask = np.isfinite(resid) & np.isfinite(denom) & (denom > 0)
    if not np.any(mask):
        return float("nan")

    k_ratio = np.abs(resid[mask]) / denom[mask]
    return float(np.nanpercentile(k_ratio, percentile))


def apply_calibration(
    median,
    lower,
    upper,
    alpha: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply an alpha scaling factor to produce calibrated bounds.
    Falls back to original bounds if alpha is invalid.
    """
    mid = _as_array(median)
    lo = _as_array(lower)
    hi = _as_array(upper)
    if mid.size == 0 or lo.size == 0 or hi.size == 0:
        return lo, hi

    if alpha is None or not np.isfinite(alpha) or alpha <= 0:
        return lo, hi

    s_hat = (hi - lo) / (2.0 * Z_90)
    adj = Z_90 * alpha * s_hat
    lo_cal = mid - adj
    hi_cal = mid + adj
    return lo_cal, hi_cal
