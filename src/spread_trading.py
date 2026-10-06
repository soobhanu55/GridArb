"""Cross-zone price-spread mean reversion: Engle-Granger cointegration, z-score signals, a cost-aware backtest
and walk-forward validation, applied to day-ahead prices of DE-LU and its neighbouring bidding zones.

This is a methodology study. Day-ahead zones are coupled by the auction itself, and there is no instrument that
lets you hold "long DE-LU, short FR" as a pure spread, so the P&L here is a paper P&L of 1 MW per leg, not a
tradable strategy. Prices go negative, so everything is in EUR/MWh differences, never percentage returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

HOURS_PER_DAY = 24  # 1 MW held all day = 24 MWh


@dataclass
class Pair:
    a: str
    b: str
    p_value: float
    beta: float  # a ~ beta * b + const
    half_life: float  # days


def hedge_ratio(a: pd.Series, b: pd.Series) -> float:
    return float(sm.OLS(a.values, sm.add_constant(b.values)).fit().params[1])


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life from an AR(1) fit on the spread's changes (inf if it does not revert)."""
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    theta = sm.OLS(delta.values, sm.add_constant(lagged.values)).fit().params[1]
    return float("inf") if theta >= 0 else float(-np.log(2) / theta)


def find_cointegrated_pairs(prices: pd.DataFrame, significance: float = 0.05) -> list[Pair]:
    """All pairs passing the Engle-Granger test, strongest first. With 21 pairs, about one passes by chance at 5%."""
    out = []
    for a, b in combinations(prices.columns, 2):
        p = coint(prices[a], prices[b])[1]
        if p < significance:
            beta = hedge_ratio(prices[a], prices[b])
            out.append(Pair(a, b, float(p), beta, half_life(prices[a] - beta * prices[b])))
    return sorted(out, key=lambda r: r.p_value)


def rolling_zscore(spread: pd.Series, window: int = 30) -> pd.Series:
    return (spread - spread.rolling(window).mean()) / spread.rolling(window).std()


def generate_signals(z: pd.Series, entry: float = 2.0, exit_: float = 0.5) -> pd.Series:
    """Position in {-1, 0, +1} (+1 = long a, short beta*b). Enter beyond +-entry, hold until |z| falls inside exit_."""
    pos, cur = [], 0
    for v in z:
        if not np.isnan(v):
            if cur == 0:
                cur = 1 if v < -entry else -1 if v > entry else 0
            elif (cur == 1 and v > -exit_) or (cur == -1 and v < exit_):
                cur = 0
        pos.append(cur)
    return pd.Series(pos, index=z.index)


def backtest(a: pd.Series, b: pd.Series, beta: float, position: pd.Series, cost_eur_mwh: float = 0.5) -> pd.Series:
    """Daily P&L in EUR. The position set on day t earns on day t+1 (no look-ahead); the spread P&L is
    (change in a - beta * change in b) x 24 MWh. Cost is charged per MWh traded on both legs when the position changes."""
    spread_change = a.diff() - beta * b.diff()
    gross = position.shift(1).fillna(0) * spread_change.fillna(0) * HOURS_PER_DAY
    traded_mwh = position.diff().abs().fillna(position.abs()) * (1 + abs(beta)) * HOURS_PER_DAY
    return gross - traded_mwh * cost_eur_mwh


def sharpe(pnl: pd.Series, periods: int = 365) -> float:
    return 0.0 if pnl.std() == 0 else float(np.sqrt(periods) * pnl.mean() / pnl.std())


def max_drawdown(pnl: pd.Series) -> float:
    """Largest peak-to-trough fall of cumulative P&L, in EUR (positive number)."""
    cum = pnl.cumsum()
    return float((cum.cummax() - cum).max())


@dataclass
class WalkForward:
    windows: list[dict] = field(default_factory=list)
    pnl: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def walk_forward(prices: pd.DataFrame, formation: int = 180, trading: int = 30, z_window: int = 30,
                 entry: float = 2.0, exit_: float = 0.5, cost_eur_mwh: float = 0.5) -> WalkForward:
    """Each window: pick the best cointegrated pair and hedge ratio on `formation` days only, then trade it
    on the next `trading` days. Windows with no cointegrated pair stay flat."""
    res, segments, start = WalkForward(), [], 0
    while start + formation + trading <= len(prices):
        form = prices.iloc[start:start + formation]
        trade = prices.iloc[start + formation:start + formation + trading]
        pairs = find_cointegrated_pairs(form)
        if pairs:
            p = pairs[0]
            both = pd.concat([form[[p.a, p.b]].tail(z_window), trade[[p.a, p.b]]])  # warm-up rows for the z-score only
            z = rolling_zscore(both[p.a] - p.beta * both[p.b], z_window)
            pos = generate_signals(z, entry, exit_)
            pnl = backtest(both[p.a], both[p.b], p.beta, pos, cost_eur_mwh).loc[trade.index]
            res.windows.append({"pair": f"{p.a}-{p.b}", "start": str(trade.index[0].date()), "pnl": float(pnl.sum()),
                                "p_value": p.p_value, "beta": p.beta, "half_life": p.half_life})
        else:
            pnl = pd.Series(0.0, index=trade.index)
            res.windows.append({"pair": None, "start": str(trade.index[0].date()), "pnl": 0.0})
        segments.append(pnl)
        start += trading
    res.pnl = pd.concat(segments) if segments else res.pnl
    return res
