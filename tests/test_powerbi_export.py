"""The Power BI export turns results.json into a consistent star schema."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def exporter():
    spec = importlib.util.spec_from_file_location("export_powerbi", ROOT / "scripts" / "export_powerbi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model(family, mae, profit):
    return {"family": family, "forecast_metrics": {"mae": mae, "rmse": mae * 1.4, "r2": 0.6}, "trading_summary": {"total_profit_eur": profit},
            "pct_of_perfect": 100 * profit / 300.0, "fit_predict_seconds": 1.5,
            "daily_profit": {"2026-04-10 00:00:00+00:00": profit / 2, "2026-04-11 00:00:00+00:00": profit / 2}}


@pytest.fixture
def results():
    ts = pd.date_range("2026-04-10", periods=48, freq="h", tz="UTC")
    return {
        "models": {"naive": model("baseline", 40.0, 280.0), "lstm": model("neural", 33.0, 270.0)},
        "perfect_foresight": {"trading_summary": {"total_profit_eur": 300.0},
                              "daily_profit": {"2026-04-10 00:00:00+00:00": 150.0, "2026-04-11 00:00:00+00:00": 150.0}},
        "chart_tail": {"timestamps": [t.isoformat() for t in ts], "actual": list(range(48)),
                       "pred_naive": [x + 1 for x in range(48)], "pred_lstm": [x - 2 for x in range(48)]},
    }


def test_keys_resolve_and_neural_flag_is_set(exporter, results):
    t = exporter.build_tables(results)
    assert set(t["fact_model_summary"]["model_key"]) == set(t["dim_model"]["model_key"])
    assert set(t["fact_model_day"]["model_key"]) <= set(t["dim_model"]["model_key"])
    assert set(t["fact_model_day"]["date_key"]) <= set(t["dim_date"]["date_key"])
    assert dict(zip(t["dim_model"]["model"], t["dim_model"]["is_neural"])) == {"naive": 0, "lstm": 1}


def test_daily_percent_of_perfect_uses_the_perfect_foresight_day(exporter, results):
    day = exporter.build_tables(results)["fact_model_day"]
    row = day[(day["model_key"] == 1) & (day["date_key"] == 20260410)].iloc[0]
    assert row["profit_eur"] == 140.0 and row["perfect_profit_eur"] == 150.0 and row["pct_of_perfect_day"] == pytest.approx(140 / 150)


def test_per_day_error_and_bias_are_computed_from_predictions(exporter, results):
    idx = pd.date_range("2026-04-10", periods=48, freq="h", tz="UTC")
    actual = pd.Series(np.full(48, 100.0), index=idx)
    preds = {"naive": pd.Series(np.full(48, 110.0), index=idx), "lstm": pd.Series(np.full(48, 95.0), index=idx)}
    day = exporter.build_tables(results, preds, actual)["fact_model_day"]
    naive = day[day["model_key"] == 1].iloc[0]
    lstm = day[day["model_key"] == 2].iloc[0]
    assert naive["mae"] == pytest.approx(10.0) and naive["bias"] == pytest.approx(10.0)
    assert lstm["mae"] == pytest.approx(5.0) and lstm["bias"] == pytest.approx(-5.0)


def test_hourly_tail_has_one_row_per_model_and_hour(exporter, results):
    h = exporter.build_tables(results)["fact_hourly_tail"]
    assert len(h) == 2 * 48 and (h["predicted"] - h["actual"]).abs().max() == 2


def test_export_writes_every_table(exporter, results, tmp_path):
    import json

    (tmp_path / "results.json").write_text(json.dumps(results))
    counts = exporter.export(tmp_path / "out", tmp_path / "results.json", tmp_path / "no_cache")
    assert set(counts) == {"dim_model", "dim_date", "fact_model_summary", "fact_model_day", "fact_hourly_tail"}
    assert (tmp_path / "out" / "fact_model_summary.csv").exists()
