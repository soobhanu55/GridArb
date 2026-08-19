import numpy as np
import pandas as pd
import pytest

from src.features import add_calendar_features, add_lag_features, build_feature_frame, GERMAN_HOLIDAYS


def _synthetic_hourly(n_hours: int, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    # price[i] = i, so lag relationships are trivially hand-checkable.
    price = np.arange(n_hours, dtype=float)
    load = np.arange(n_hours, dtype=float) * 10
    return pd.DataFrame({"price_eur_mwh": price, "load_mw": load}, index=idx)


def test_calendar_features_hour_and_dayofweek():
    df = _synthetic_hourly(48, start="2024-01-01")  # 2024-01-01 is a Monday
    out = add_calendar_features(df)
    assert out["hour"].iloc[0] == 0
    assert out["hour"].iloc[5] == 5
    assert out["dayofweek"].iloc[0] == 0  # Monday
    assert out["is_weekend"].iloc[0] == 0
    # 2024-01-06 would be Saturday; check a Saturday timestamp within range if present
    sat = pd.Timestamp("2024-01-06 12:00", tz="UTC")
    df2 = _synthetic_hourly(24 * 10, start="2024-01-01")
    out2 = add_calendar_features(df2)
    assert out2.loc[sat, "is_weekend"] == 1


def test_calendar_features_known_holiday():
    df = _synthetic_hourly(24, start="2024-01-01")  # New Year's Day, a known German holiday
    out = add_calendar_features(df)
    assert (out["is_holiday"] == 1).all()
    assert pd.Timestamp("2024-01-01").date() in GERMAN_HOLIDAYS


def test_calendar_cyclical_encoding_hour_0_and_23_are_adjacent():
    """hour=0 and hour=23 should be close in sin/cos space (they're adjacent
    on the clock), unlike raw integer encoding where they're maximally far
    apart (0 vs 23).
    """
    df = _synthetic_hourly(24, start="2024-01-01")
    out = add_calendar_features(df)
    h0 = out.iloc[0][["hour_sin", "hour_cos"]].to_numpy()
    h23 = out.iloc[23][["hour_sin", "hour_cos"]].to_numpy()
    h12 = out.iloc[12][["hour_sin", "hour_cos"]].to_numpy()
    dist_0_23 = np.linalg.norm(h0 - h23)
    dist_0_12 = np.linalg.norm(h0 - h12)
    assert dist_0_23 < dist_0_12  # midnight-to-11pm closer than midnight-to-noon


def test_lag_features_hand_calculated_no_leakage():
    """price[i] = i by construction, so price_lag_24h at row t must equal
    exactly t-24, and critically must never equal price[t] itself (the
    thing a leaking implementation would accidentally expose).
    """
    df = _synthetic_hourly(300)
    out = add_lag_features(df)

    row = out.iloc[250]  # price_eur_mwh at this row == 250
    assert row["price_eur_mwh"] == 250
    assert row["price_lag_24h"] == 250 - 24
    assert row["price_lag_48h"] == 250 - 48
    assert row["price_lag_168h"] == 250 - 168
    assert row["price_lag_24h"] != row["price_eur_mwh"]


def test_rolling_mean_excludes_current_and_last_24h():
    """price_roll_mean_24h at row t is the mean of price[t-48..t-25]
    (a 24h window ending at t-25, since the rolling series itself is built
    on price shifted by 24). Hand-check against a direct slice.
    """
    df = _synthetic_hourly(300)
    out = add_lag_features(df)

    t = 250
    expected_window = np.arange(t - 24 - 23, t - 24 + 1)  # price[t-47..t-24]
    expected_mean = expected_window.mean()
    assert out.iloc[t]["price_roll_mean_24h"] == pytest.approx(expected_mean, abs=1e-9)


def test_build_feature_frame_drops_warmup_rows_and_has_no_nans():
    df = _synthetic_hourly(24 * 20)  # 20 days, enough for 168h lookback + buffer
    feat = build_feature_frame(df)
    assert len(feat) < len(df)  # warmup rows (first ~168h) dropped
    feature_cols = [c for c in feat.columns if c not in ("price_eur_mwh", "load_mw")]
    assert not feat[feature_cols + ["price_eur_mwh"]].isna().any().any()


def test_build_feature_frame_never_leaks_target_into_its_own_features():
    """Every lag/rolling feature column at row t must be strictly less than
    price_eur_mwh at row t, since price is monotonically increasing by
    construction (price[i] = i) -- any feature >= the target would mean it
    saw same-or-future information.
    """
    df = _synthetic_hourly(24 * 20)
    feat = build_feature_frame(df)
    lag_cols = [c for c in feat.columns if c.startswith("price_lag_") or c.startswith("price_roll_")]
    for col in lag_cols:
        assert (feat[col] < feat["price_eur_mwh"]).all(), f"{col} leaks future information"
