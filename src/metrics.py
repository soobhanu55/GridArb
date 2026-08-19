"""Forecast accuracy metrics.

MAPE is deliberately omitted: German day-ahead prices go negative and pass
through near-zero (renewable oversupply hours), which makes MAPE blow up
or divide by ~0 and produce meaningless numbers. MAE/RMSE in EUR/MWh and R^2
are stable and directly interpretable regardless of sign.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
    }
