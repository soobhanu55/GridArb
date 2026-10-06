"""Walk-forward spread study on real DE-LU and neighbouring-zone day-ahead prices (writes docs/spread_study.md).

    python scripts/spread_study.py

Compares a naive run (pair chosen on the whole history, then traded on that same history; look-ahead by
construction) with walk-forward validation, at three cost levels, and with the number of pairs tested stated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_zone_prices  # noqa: E402
from src.spread_trading import (backtest, find_cointegrated_pairs, generate_signals, max_drawdown,  # noqa: E402
                                rolling_zscore, sharpe, walk_forward)

COSTS = (0.0, 0.5, 1.0)  # EUR per MWh traded


def naive(prices: pd.DataFrame, cost: float) -> tuple[str, pd.Series]:
    p = find_cointegrated_pairs(prices)[0]
    z = rolling_zscore(prices[p.a] - p.beta * prices[p.b])
    return f"{p.a}-{p.b}", backtest(prices[p.a], prices[p.b], p.beta, generate_signals(z), cost).iloc[30:]


def bootstrap_ci(pnl: pd.Series, n: int = 2000, block: int = 10, seed: int = 0) -> tuple[float, float]:
    """95% interval for the mean daily P&L, moving-block bootstrap (daily P&L is autocorrelated while a trade is open)."""
    rng = np.random.default_rng(seed)
    x, k = pnl.values, int(np.ceil(len(pnl) / block))
    means = [np.concatenate([x[s:s + block] for s in rng.integers(0, len(x) - block, k)])[:len(x)].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    hourly = load_zone_prices()
    daily = hourly.resample("D").mean().dropna()
    zones = list(daily.columns)
    n_pairs = len(zones) * (len(zones) - 1) // 2
    lines = [
        "# Cross-zone spread study (walk-forward)\n",
        f"Day-ahead prices of {', '.join(zones)}, daily means, {daily.index[0].date()} to {daily.index[-1].date()} "
        f"({len(daily)} days). {n_pairs} candidate pairs; at a 5% Engle-Granger level about {n_pairs * 0.05:.1f} pass by chance "
        "in any window. Paper P&L of 1 MW per leg (24 MWh per day), costs per MWh traded on both legs. "
        "Day-ahead zones are coupled by the auction, so this is a methodology study, not a tradable strategy.\n",
        "| Cost (EUR/MWh) | Method | Total P&L (EUR) | Sharpe | Max drawdown (EUR) | Mean daily P&L, 95% CI (EUR) |",
        "|---|---|---|---|---|---|",
    ]
    wf_ref = None
    for cost in COSTS:
        wf = walk_forward(daily, cost_eur_mwh=cost)
        wf_ref = wf_ref or wf
        pair, nv = naive(daily, cost)
        for label, pnl in ((f"naive ({pair}, look-ahead)", nv), ("walk-forward", wf.pnl)):
            lo, hi = bootstrap_ci(pnl)
            lines.append(f"| {cost} | {label} | {pnl.sum():,.0f} | {sharpe(pnl):.2f} | {max_drawdown(pnl):,.0f} | {pnl.mean():.0f} ({lo:.0f} to {hi:.0f}) |")

    chosen = pd.Series([w["pair"] or "none" for w in wf_ref.windows]).value_counts()
    active = [w for w in wf_ref.windows if w["pair"]]
    lines += [
        "", f"Walk-forward: {len(wf_ref.windows)} windows (180 days formation, 30 days trading), {len(active)} with a cointegrated pair. "
        f"Pairs chosen: {', '.join(f'{k} x{v}' for k, v in chosen.items())}. "
        f"Windows with positive P&L at cost 0: {sum(w['pnl'] > 0 for w in active)} of {len(active)}.",
        "",
        "**Do not read the Sharpe ratios as an edge.** Market coupling keeps neighbouring zones' prices tied together, so the screen found a pair in every window; "
        "the spreads are stationary, noisy series that always snap back; the strategy collects "
        "that reversion. No instrument lets you enter at one day's auction price and exit at the next day's, so this P&L cannot be "
        "realised; a real version needs a tradable spread product, margin and execution rules. Two things the study does show: "
        "walk-forward selection costs little against the look-ahead run here (the spread is stationary whichever pair is picked), "
        "and, unlike PairForge's original sector-ETF universe where nothing survived costs, a structurally coupled universe "
        "makes the screen find a pair in every window.",
    ]
    out = ROOT / "docs" / "spread_study.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
