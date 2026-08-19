import numpy as np
import pytest

from src.battery import BatteryConfig, dispatch_profit, optimize_dispatch


def test_lossless_charge_low_discharge_high():
    """Hand-calculated: cheap-cheap-expensive-expensive, no losses, cap=2MWh,
    power=1MW, start empty. Optimal is fully charge on the two cheap hours,
    fully discharge on the two expensive hours.
    """
    prices = np.array([10.0, 10.0, 100.0, 100.0])
    cfg = BatteryConfig(capacity_mwh=2.0, max_power_mw=1.0, round_trip_efficiency=1.0, initial_soc_mwh=0.0)
    r = optimize_dispatch(prices, cfg)

    assert r["charge"] == pytest.approx([1.0, 1.0, 0.0, 0.0], abs=1e-6)
    assert r["discharge"] == pytest.approx([0.0, 0.0, 1.0, 1.0], abs=1e-6)
    assert r["soc"] == pytest.approx([1.0, 2.0, 1.0, 0.0], abs=1e-6)

    profit = dispatch_profit(r["charge"], r["discharge"], prices)
    assert profit == pytest.approx(180.0, abs=1e-6)  # -10-10+100+100


def test_flat_prices_no_incentive_to_trade():
    """No price spread -- any round-trip trade is a wash at best (a loss
    with any real efficiency loss), so the optimum is to do nothing at all.
    Starts empty deliberately: a non-zero initial SoC would be "free" stored
    energy the LP can profitably discharge regardless of price flatness,
    which would defeat the point of this test.
    """
    prices = np.full(6, 50.0)
    cfg = BatteryConfig(capacity_mwh=2.0, max_power_mw=1.0, round_trip_efficiency=0.9, initial_soc_mwh=0.0)
    r = optimize_dispatch(prices, cfg)

    profit = dispatch_profit(r["charge"], r["discharge"], prices)
    assert profit == pytest.approx(0.0, abs=1e-6)


def test_nonzero_initial_soc_is_profitably_drained_even_at_flat_prices():
    """Sanity check for the above: if we DO start with stored energy, the
    LP correctly treats it as free and drains it for profit even when
    there's no price spread to exploit -- this is what test_flat_prices_
    no_incentive_to_trade deliberately avoids by starting empty.
    """
    prices = np.full(6, 50.0)
    cfg = BatteryConfig(capacity_mwh=2.0, max_power_mw=1.0, round_trip_efficiency=0.9, initial_soc_mwh=1.0)
    r = optimize_dispatch(prices, cfg)

    profit = dispatch_profit(r["charge"], r["discharge"], prices)
    eta = np.sqrt(0.9)
    expected = 1.0 * eta * 50.0  # fully discharge the free initial 1.0 MWh
    assert profit == pytest.approx(expected, abs=1e-6)


def test_round_trip_efficiency_reduces_deliverable_energy():
    """With 81% round-trip efficiency (eta=0.9 per leg), charging 2MWh in
    only leaves 2*0.9=1.8MWh stored, and discharging that back out loses
    another 10%, so realized profit must be strictly less than the lossless
    case's 180, but still positive since 100 >> 10.
    """
    prices = np.array([10.0, 10.0, 100.0, 100.0])
    cfg = BatteryConfig(capacity_mwh=2.0, max_power_mw=1.0, round_trip_efficiency=0.81, initial_soc_mwh=0.0)
    r = optimize_dispatch(prices, cfg)

    profit = dispatch_profit(r["charge"], r["discharge"], prices)
    assert 0 < profit < 180.0
    assert profit == pytest.approx(142.0, abs=1e-6)

    # Energy conservation: total discharged must equal eta^2 * total charged.
    total_charged = r["charge"].sum()
    total_discharged = r["discharge"].sum()
    assert total_discharged == pytest.approx(total_charged * 0.81, abs=1e-6)


def test_capacity_constraint_respected():
    """Even with unlimited cheap hours, SoC can never exceed capacity."""
    prices = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    cfg = BatteryConfig(capacity_mwh=1.5, max_power_mw=1.0, round_trip_efficiency=1.0, initial_soc_mwh=0.0)
    r = optimize_dispatch(prices, cfg)

    assert np.all(r["soc"] <= 1.5 + 1e-6)
    assert np.all(r["soc"] >= -1e-6)


def test_dispatch_profit_uses_settlement_prices_not_decision_prices():
    """A schedule decided on one price series can be settled against a
    different one -- this is the whole point of forecast-based dispatch
    (decide blind, settle against reality).
    """
    charge = np.array([1.0, 0.0])
    discharge = np.array([0.0, 1.0])
    settlement = np.array([5.0, 50.0])
    profit = dispatch_profit(charge, discharge, settlement)
    assert profit == pytest.approx(45.0, abs=1e-6)  # -5 + 50
