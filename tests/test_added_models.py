"""Tests for the model families absorbed from ForecastNet / DemandLens."""
import numpy as np
import pandas as pd
import pytest

from src.features import _wavelet_denoise_last, add_wavelet_features
from src.models import (CNNLSTMModel, CNNModel, HoltWintersModel, LSTMModel, MLPModel,
                        RandomForestModel, SARIMAModel)


def _prices(n_hours: int, noise: float = 0.0, seed: int = 0) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    t = np.arange(n_hours)
    base = 50 + 10 * np.sin(t * 2 * np.pi / 24) + 5 * np.sin(t * 2 * np.pi / 168)
    return pd.Series(base + np.random.default_rng(seed).normal(0, noise, n_hours), index=idx)


# ---- wavelet feature ------------------------------------------------------

def test_wavelet_feature_is_causal():
    """Changing prices at/after t-23h must not change the feature at t."""
    base = pd.DataFrame({"price_eur_mwh": _prices(500, noise=2.0)})
    t_pos = 400
    changed = base.copy()
    changed.iloc[t_pos - 23:, 0] += 1000.0  # everything the day-ahead cutoff forbids
    a = add_wavelet_features(base)["price_wavelet_24h"].iloc[t_pos]
    b = add_wavelet_features(changed)["price_wavelet_24h"].iloc[t_pos]
    assert a == b


def test_wavelet_feature_denoises():
    clean, noisy = _prices(600), _prices(600, noise=5.0)
    out = add_wavelet_features(pd.DataFrame({"price_eur_mwh": noisy}))["price_wavelet_24h"]
    lagged_clean = clean.shift(24)
    ok = out.notna()
    err_denoised = np.abs(out[ok] - lagged_clean[ok]).mean()
    err_raw = np.abs(noisy.shift(24)[ok] - lagged_clean[ok]).mean()
    assert err_denoised < err_raw


def test_wavelet_denoise_accepts_read_only_input():
    """pandas copy-on-write hands out read-only arrays; CI once failed with 'buffer source array is read-only'."""
    window = np.random.default_rng(0).normal(size=168)
    window.setflags(write=False)
    assert np.isfinite(_wavelet_denoise_last(window, "db4", 3))


def test_wavelet_feature_nan_during_warmup():
    out = add_wavelet_features(pd.DataFrame({"price_eur_mwh": _prices(300)}))["price_wavelet_24h"]
    assert out.iloc[:167 + 24].isna().all() and out.iloc[-1] == out.iloc[-1]


# ---- tabular + sequence models ---------------------------------------------

def test_random_forest_learns_a_simple_rule():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = 3 * X["a"]
    model = RandomForestModel(n_estimators=50).fit(X, y)
    assert np.corrcoef(model.predict(X), y)[0, 1] > 0.9


@pytest.mark.parametrize("cls", [LSTMModel, MLPModel, CNNModel, CNNLSTMModel])
def test_sequence_models_fit_predict_shape_no_nans(cls):
    price = _prices(24 * 15)
    cut = len(price) - 24
    train_idx, test_idx = price.index[168:cut], price.index[cut:]
    model = cls(lookback=48, epochs=2)
    model.fit(pd.DataFrame(index=train_idx), price.loc[train_idx], price_history=price.iloc[:cut])
    preds = model.predict(pd.DataFrame(index=test_idx), price_history=price)
    assert preds.shape == (24,) and not np.isnan(preds).any()


def test_sequence_models_are_seeded_and_reproducible():
    price = _prices(24 * 12)
    cut = len(price) - 24
    runs = []
    for _ in range(2):
        m = MLPModel(lookback=48, epochs=2, seed=7)
        m.fit(pd.DataFrame(index=price.index[168:cut]), price.iloc[168:cut], price_history=price.iloc[:cut])
        runs.append(m.predict(pd.DataFrame(index=price.index[cut:]), price_history=price))
    np.testing.assert_allclose(runs[0], runs[1], rtol=1e-5)


# ---- classical models -------------------------------------------------------

@pytest.mark.parametrize("cls", [HoltWintersModel, SARIMAModel])
def test_classical_models_forecast_the_requested_timestamps(cls):
    price = _prices(24 * 40)
    train, test = price.iloc[:-48], price.iloc[-48:]
    model = cls().fit(pd.DataFrame(index=train.index), train)
    full = model.predict(pd.DataFrame(index=test.index))
    assert full.shape == (48,) and not np.isnan(full).any()
    # asking for only the second day must return that day's slice of the same forecast
    second_day = model.predict(pd.DataFrame(index=test.index[24:]))
    np.testing.assert_allclose(second_day, full[24:])


def test_holt_winters_beats_flat_mean_on_seasonal_data():
    price = _prices(24 * 40)
    train, test = price.iloc[:-168], price.iloc[-168:]
    preds = HoltWintersModel().fit(pd.DataFrame(index=train.index), train).predict(pd.DataFrame(index=test.index))
    assert np.abs(preds - test.to_numpy()).mean() < np.abs(train.mean() - test.to_numpy()).mean()
