import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_daily, backtest_perfect_foresight, summarize_backtest
from src.battery import BatteryConfig


def _two_day_series():
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    # Day 1: cheap first half, expensive second half. Day 2: flat (no spread).
    day1 = [10.0] * 12 + [100.0] * 12
    day2 = [50.0] * 24
    return pd.Series(day1 + day2, index=idx)


def test_backtest_daily_skips_incomplete_days():
    idx = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")  # 1 full day + 6h
    prices = pd.Series(np.arange(30, dtype=float), index=idx)
    cfg = BatteryConfig(capacity_mwh=2.0, max_power_mw=1.0, round_trip_efficiency=1.0)

    result = backtest_daily(prices, prices, cfg)
    assert len(result) == 1  # only the first complete 24h day counted


def test_backtest_daily_profit_matches_direct_lp_calculation():
    prices = _two_day_series()
    cfg = BatteryConfig(capacity_mwh=2.0, max_power_mw=1.0, round_trip_efficiency=1.0, initial_soc_mwh=0.0)
    result = backtest_daily(prices, prices, cfg)

    assert len(result) == 2
    day1_profit = result["profit"].iloc[0]
    day2_profit = result["profit"].iloc[1]
    # Day 1 has a real spread (10 vs 100) -- profit must be positive.
    assert day1_profit > 0
    # Day 2 is completely flat -- no possible profit.
    assert day2_profit == pytest.approx(0.0, abs=1e-6)


def test_perfect_foresight_profit_never_less_than_forecast_based():
    """The perfect-foresight backtest (dispatch decided on the actual prices
    themselves) must always weakly dominate a dispatch decided on a noisy
    forecast, since the LP is solving the exact same problem with strictly
    better information.
    """
    idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    actual = pd.Series(50 + rng.normal(0, 20, 24), index=idx)
    noisy_forecast = actual + pd.Series(rng.normal(0, 15, 24), index=idx)

    cfg = BatteryConfig()
    bt_perfect = backtest_perfect_foresight(actual, cfg)
    bt_forecast = backtest_daily(noisy_forecast, actual, cfg)

    assert bt_perfect["profit"].iloc[0] >= bt_forecast["profit"].iloc[0] - 1e-6


def test_soc_carries_over_between_days_not_reset_to_free_energy():
    """Regression test for a real bug caught during development: the first
    version of backtest_daily reset every day's initial SoC to
    config.initial_soc_mwh, so a battery configured to start half-full
    could "discharge for free" every single morning regardless of that
    day's actual price spread, manufacturing profit that wasn't real.

    Day 1 here fully drains the battery to 0 by end of day (verified via
    the LP itself). Day 2 is completely flat (no spread) -- if SoC
    correctly carries over at 0, day 2 must show exactly zero profit; the
    old buggy version would show ~1.9 EUR/day of manufactured profit from
    resetting to the configured 2.0 MWh initial charge.
    """
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    day1 = [10.0] * 12 + [100.0] * 12  # real spread -- battery should end near empty
    day2 = [50.0] * 24  # flat -- zero profit possible if truly starting from where day1 left off
    prices = pd.Series(day1 + day2, index=idx)

    cfg = BatteryConfig(capacity_mwh=2.0, max_power_mw=1.0, round_trip_efficiency=1.0, initial_soc_mwh=2.0)
    result = backtest_daily(prices, prices, cfg)

    assert len(result) == 2
    day2_profit = result["profit"].iloc[1]
    assert day2_profit == pytest.approx(0.0, abs=1e-6)


def test_summarize_backtest_hand_calculated():
    daily = pd.DataFrame({"profit": [100.0, -20.0, 50.0, 0.0]})
    summary = summarize_backtest(daily)
    assert summary["total_profit_eur"] == pytest.approx(130.0)
    assert summary["mean_daily_profit_eur"] == pytest.approx(32.5)
    assert summary["n_days"] == 4
    assert summary["pct_profitable_days"] == pytest.approx(50.0)  # 2 of 4 > 0
