"""Battery storage arbitrage: given a set of hourly prices, find the
charge/discharge schedule that maximizes profit, subject to real battery
constraints (capacity, power limit, round-trip efficiency, state of charge).

Solved as a linear program per dispatch window (typically 24h) via
scipy.optimize.linprog. This mirrors how day-ahead battery dispatch is
actually optimized in practice: decide the full day's schedule once, using
the day-ahead price forecast, before any of that day's hours happen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class BatteryConfig:
    capacity_mwh: float = 4.0
    max_power_mw: float = 1.0
    round_trip_efficiency: float = 0.9
    initial_soc_mwh: float = 2.0


def optimize_dispatch(prices: np.ndarray, config: BatteryConfig) -> dict:
    """Returns {charge, discharge, soc} arrays (length = len(prices)) that
    maximize sum(discharge[h]*price[h] - charge[h]*price[h]) subject to:
      - 0 <= charge[h], discharge[h] <= max_power_mw
      - 0 <= soc[h] <= capacity_mwh
      - soc[h] = soc[h-1] + charge[h]*sqrt(eta) - discharge[h]/sqrt(eta)
        (efficiency split symmetrically across charge/discharge legs)
    """
    n = len(prices)
    eta = np.sqrt(config.round_trip_efficiency)

    # Decision vector x = [charge_0..n-1, discharge_0..n-1], length 2n.
    # linprog minimizes, so negate the profit objective.
    c = np.concatenate([prices, -prices])  # minimize (price*charge - price*discharge)

    # Equality constraints encode SoC recursion via cumulative sums:
    # soc[h] = initial_soc + eta*sum(charge[0..h]) - (1/eta)*sum(discharge[0..h])
    # Rather than adding soc as separate variables, bound the cumulative
    # sums directly so 0 <= soc[h] <= capacity for every h.
    A_ub = []
    b_ub = []
    for h in range(n):
        row_upper = np.zeros(2 * n)
        row_upper[: h + 1] = eta
        row_upper[n: n + h + 1] = -1.0 / eta
        A_ub.append(row_upper)
        b_ub.append(config.capacity_mwh - config.initial_soc_mwh)

        row_lower = -row_upper
        A_ub.append(row_lower)
        b_ub.append(config.initial_soc_mwh)

    bounds = [(0, config.max_power_mw)] * (2 * n)

    result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Battery dispatch LP failed: {result.message}")

    charge = result.x[:n]
    discharge = result.x[n:]
    soc = np.empty(n)
    running = config.initial_soc_mwh
    for h in range(n):
        running += eta * charge[h] - discharge[h] / eta
        soc[h] = running

    return {"charge": charge, "discharge": discharge, "soc": soc}


def dispatch_profit(charge: np.ndarray, discharge: np.ndarray, settlement_prices: np.ndarray) -> float:
    """P&L of executing a given charge/discharge schedule against the prices
    it actually settles at (which may differ from the prices used to decide
    the schedule, if the schedule was chosen from a forecast).
    """
    return float(np.sum(discharge * settlement_prices - charge * settlement_prices))
