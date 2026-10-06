"""Feature engineering for hourly day-ahead electricity price forecasting.

All features are computed causally: at time t, only information available
up to and including t is used (no leakage from the future). Lag/rolling
features are shifted so that a training row for hour t never sees price[t]
itself as a predictor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# German public holidays (national) for the years covered by the dataset.
# Hand-maintained rather than pulling a holiday library, since this is a
# short, fixed list and avoids an extra dependency for ~15 dates/year.
GERMAN_HOLIDAYS = pd.to_datetime([
    "2023-01-01", "2023-04-07", "2023-04-10", "2023-05-01", "2023-05-18",
    "2023-05-29", "2023-10-03", "2023-12-25", "2023-12-26",
    "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-01", "2024-05-09",
    "2024-05-20", "2024-10-03", "2024-12-25", "2024-12-26",
    "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01", "2025-05-29",
    "2025-06-09", "2025-10-03", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-01", "2026-05-14",
    "2026-05-25", "2026-10-03", "2026-12-25", "2026-12-26",
]).date


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_holiday"] = pd.Series(idx.date, index=idx).isin(GERMAN_HOLIDAYS).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(df: pd.DataFrame, price_col: str = "price_eur_mwh") -> pd.DataFrame:
    """Lag/rolling features built only from information at or before hour t-24,
    since a day-ahead forecast made "now" cannot see the last 24h of realized
    price for the day it is forecasting (that's the whole point of day-ahead).
    """
    df = df.copy()
    p = df[price_col]

    for lag_hours in (24, 48, 168):  # yesterday, 2 days ago, same hour last week
        df[f"price_lag_{lag_hours}h"] = p.shift(lag_hours)

    # Rolling stats computed on data available at least 24h before t, so a
    # window ending at t-24h -- never touches the target day's own prices.
    shifted = p.shift(24)
    df["price_roll_mean_24h"] = shifted.rolling(24).mean()
    df["price_roll_std_24h"] = shifted.rolling(24).std()
    df["price_roll_mean_168h"] = shifted.rolling(168).mean()

    if "load_mw" in df.columns:
        load_shifted = df["load_mw"].shift(24)
        df["load_lag_24h"] = load_shifted
        df["load_roll_mean_168h"] = load_shifted.rolling(168).mean()

    return df


def _wavelet_denoise_last(window: np.ndarray, wavelet: str, level: int) -> float:
    """Soft-threshold the detail coefficients (universal threshold) and return the
    denoised value at the END of the window -- the value a forecaster would act on."""
    import pywt

    coeffs = pywt.wavedec(window, wavelet, level=level, mode="symmetric")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # noise level from the finest scale
    thresh = sigma * np.sqrt(2 * np.log(len(window)))
    coeffs[1:] = [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]
    return float(pywt.waverec(coeffs, wavelet, mode="symmetric")[len(window) - 1])


def add_wavelet_features(
    df: pd.DataFrame, price_col: str = "price_eur_mwh", window: int = 168, wavelet: str = "db4", level: int = 3
) -> pd.DataFrame:
    """Wavelet-denoised price at t-24h (idea absorbed from DemandLens).

    Causal on purpose: wavelet-denoising the whole series at once would let
    future prices shape every value. Here each row denoises only the trailing
    `window` hours ending at t-24h, so the day-ahead information cutoff holds.
    """
    df = df.copy()
    values = df[price_col].shift(24).to_numpy()
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        w = values[i - window + 1 : i + 1]
        if not np.isnan(w).any():
            out[i] = _wavelet_denoise_last(w, wavelet, level)
    df["price_wavelet_24h"] = out
    return df


def build_feature_frame(raw: pd.DataFrame, price_col: str = "price_eur_mwh", wavelet: bool = False) -> pd.DataFrame:
    """Full feature pipeline: calendar + lag/rolling, target = price_eur_mwh.
    Drops rows where lag features are still NaN (the first ~168h of history).
    With wavelet=True, also adds price_wavelet_24h (costs ~another week of warm-up rows).
    """
    df = add_calendar_features(raw)
    df = add_lag_features(df, price_col=price_col)
    if wavelet:
        df = add_wavelet_features(df, price_col=price_col)
    feature_cols = [c for c in df.columns if c not in (price_col, "load_mw")]
    df = df.dropna(subset=feature_cols + [price_col])
    return df


FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_holiday",
    "price_lag_24h", "price_lag_48h", "price_lag_168h",
    "price_roll_mean_24h", "price_roll_std_24h", "price_roll_mean_168h",
    "load_lag_24h", "load_roll_mean_168h",
]

FEATURE_COLUMNS_WAVELET = FEATURE_COLUMNS + ["price_wavelet_24h"]
