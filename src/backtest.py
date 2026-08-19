"""Day-ahead battery arbitrage backtest.

For each calendar day in the test period: the battery's charge/discharge
schedule for that day is decided using a price forecast (made without
seeing that day's actual prices), then the resulting profit is evaluated
against what those hours actually settled at -- exactly how a real
day-ahead battery strategy is judged (you commit to a schedule "blind",
then the market clears at its own price).

Day boundaries are UTC calendar days. Germany's real day-ahead auction
runs on CET/CEST calendar days, so this is a simplification -- noted in
the README rather than silently glossed over.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from src.battery import BatteryConfig, dispatch_profit, optimize_dispatch


def backtest_daily(
    forecast: pd.Series,
    actual: pd.Series,
    config: BatteryConfig,
) -> pd.DataFrame:
    """Runs one full day-ahead dispatch decision per UTC calendar day.

    forecast: predicted price series (what the dispatch decision is based on)
    actual: realized price series, same index as forecast (what P&L settles at)

    Returns a DataFrame indexed by date with columns: forecast_profit
    (P&L using the forecast-based schedule, settled at actual prices) and
    n_hours (24 for a complete day; incomplete days are skipped).
    """
    df = pd.DataFrame({"forecast": forecast, "actual": actual}).dropna()
    df["date"] = df.index.date

    rows = []
    # SoC carries over day to day -- only the very first day starts at
    # config.initial_soc_mwh. Resetting to a fixed SoC every single day
    # would let the LP "discharge for free" each morning regardless of that
    # day's real price spread, manufacturing profit that isn't real.
    running_soc = config.initial_soc_mwh
    for date, day_df in sorted(df.groupby("date")):
        if len(day_df) != 24:
            continue  # skip incomplete days (edges of the dataset)
        day_df = day_df.sort_index()
        day_config = dataclasses.replace(config, initial_soc_mwh=running_soc)
        dispatch = optimize_dispatch(day_df["forecast"].to_numpy(), day_config)
        profit = dispatch_profit(dispatch["charge"], dispatch["discharge"], day_df["actual"].to_numpy())
        rows.append({"date": date, "profit": profit, "n_hours": 24})
        running_soc = float(dispatch["soc"][-1])

    return pd.DataFrame(rows).set_index("date")


def backtest_perfect_foresight(actual: pd.Series, config: BatteryConfig) -> pd.DataFrame:
    """Upper bound: dispatch decided with full knowledge of that day's actual
    prices (impossible in reality, but the ceiling any forecast-based
    strategy is bounded by).
    """
    return backtest_daily(actual, actual, config)


def summarize_backtest(daily_profits: pd.DataFrame) -> dict:
    profit = daily_profits["profit"]
    return {
        "total_profit_eur": float(profit.sum()),
        "mean_daily_profit_eur": float(profit.mean()),
        "std_daily_profit_eur": float(profit.std()),
        "n_days": int(len(profit)),
        "pct_profitable_days": float((profit > 0).mean() * 100),
    }
