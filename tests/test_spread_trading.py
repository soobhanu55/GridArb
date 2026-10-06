import numpy as np
import pandas as pd

from src.spread_trading import (backtest, find_cointegrated_pairs, generate_signals, hedge_ratio, max_drawdown,
                                walk_forward)


def _cointegrated(n=400, seed=0):
    rng = np.random.default_rng(seed)
    b = np.cumsum(rng.normal(size=n)) + 50
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i - 1] + rng.normal(scale=0.5)
    return pd.DataFrame({"A": 2 * b + noise, "B": b, "C": np.cumsum(rng.normal(size=n))},
                        index=pd.date_range("2024-01-01", periods=n, freq="D"))


def test_finds_the_cointegrated_pair_and_recovers_hedge_ratio():
    df = _cointegrated()
    pairs = find_cointegrated_pairs(df)
    assert (pairs[0].a, pairs[0].b) == ("A", "B")
    assert abs(hedge_ratio(df["A"], df["B"]) - 2) < 0.1
    assert pairs[0].half_life < 5


def test_random_walks_are_not_cointegrated_at_a_tight_threshold():
    rng = np.random.default_rng(1)
    walks = pd.DataFrame({k: np.cumsum(rng.normal(size=400)) for k in "XYZ"})
    assert find_cointegrated_pairs(walks, significance=0.001) == []


def test_signals_enter_beyond_threshold_and_hold_until_exit():
    z = pd.Series([0, -2.5, -1.5, -0.6, -0.4, 0, 2.5, 1.0, 0.4])
    assert list(generate_signals(z)) == [0, 1, 1, 1, 0, 0, -1, -1, 0]


def test_backtest_hand_calculated():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    a, b = pd.Series([10.0, 10.0, 15.0], index=idx), pd.Series([5.0, 5.0, 5.0], index=idx)
    pos = pd.Series([0, 1, 1], index=idx)  # long from day 1; spread rises 5 on day 2
    pnl = backtest(a, b, beta=1.0, position=pos, cost_eur_mwh=0.5)
    # day 1: entry costs 1 x (1+1) x 24 MWh x 0.5 = 24; day 2: earns 5 x 24 = 120
    assert list(pnl) == [0.0, -24.0, 120.0]


def test_max_drawdown():
    assert max_drawdown(pd.Series([5.0, -8.0, 2.0, -1.0])) == 8.0


def test_walk_forward_ignores_prices_after_each_window():
    df = _cointegrated(n=450)
    cut = str(df.index[300].date())
    base = walk_forward(df, formation=150, trading=30)
    changed = df.copy()
    changed.iloc[300:] = changed.iloc[300:] * 3 + 100  # tamper with the future
    after = walk_forward(changed, formation=150, trading=30)
    before_base = [w["pnl"] for w in base.windows if w["start"] < cut]
    before_after = [w["pnl"] for w in after.windows if w["start"] < cut]
    assert before_base and before_base == before_after
