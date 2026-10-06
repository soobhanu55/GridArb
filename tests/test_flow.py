"""The Prefect flow on a synthetic price series: same numbers as the plain pipeline, failures isolated, retries work."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("prefect")

from prefect.testing.utilities import prefect_test_harness  # noqa: E402

from flows import gridarb_flow  # noqa: E402
from src import pipeline  # noqa: E402
from src.features import build_feature_frame  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def prefect_backend():
    with prefect_test_harness():
        yield


@pytest.fixture(scope="module")
def prices():
    idx = pd.date_range("2025-01-01", periods=24 * 100, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    hour, day = idx.hour.to_numpy(), np.arange(len(idx)) // 24
    price = 80 + 40 * np.sin(2 * np.pi * (hour - 6) / 24) + 10 * np.sin(2 * np.pi * day / 7) + rng.normal(0, 5, len(idx))
    return pd.DataFrame({"price_eur_mwh": price, "load_mw": 50_000 + 8_000 * np.sin(2 * np.pi * hour / 24)}, index=idx)


def run(prices, **kw):
    return gridarb_flow.gridarb_pipeline(prices=prices, test_days=14, retrain_every_days=7, cache_dir=None, out_path=None,
                                         with_shap=False, track_mlflow=False, **kw)


def test_flow_matches_the_plain_pipeline(prices):
    out = run(prices, models=["naive", "linear"])
    feat = build_feature_frame(prices, wavelet=True)
    preds, metrics, _ = pipeline.run_model("linear", feat, prices["price_eur_mwh"], 14, 7, cache_dir=None)
    assert out["models"]["linear"]["forecast_metrics"] == metrics
    assert set(out["models"]) == {"naive", "linear"} and out["failed_models"] == {}
    assert out["perfect_foresight"]["trading_summary"]["total_profit_eur"] >= out["models"]["linear"]["trading_summary"]["total_profit_eur"]
    assert 0 < out["models"]["linear"]["pct_of_perfect"] <= 100 + 1e-6


def test_one_failing_model_does_not_discard_the_others(prices, monkeypatch):
    real = pipeline.run_model

    def flaky(name, *a, **k):
        if name == "linear":
            raise ValueError("boom")
        return real(name, *a, **k)

    monkeypatch.setattr(pipeline, "run_model", flaky)
    out = run(prices, models=["naive", "linear"])
    assert set(out["models"]) == {"naive"} and "ValueError: boom" in out["failed_models"]["linear"]


def test_all_models_failing_raises(prices, monkeypatch):
    monkeypatch.setattr(pipeline, "run_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError, match="every model failed"):
        run(prices, models=["naive"])


def test_forecast_task_retries_once_then_succeeds(prices, monkeypatch):
    calls = {"n": 0}
    real = pipeline.run_model

    def once_flaky(name, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient")
        return real(name, *a, **k)

    monkeypatch.setattr(pipeline, "run_model", once_flaky)
    monkeypatch.setattr(gridarb_flow.forecast, "retry_delay_seconds", [0])
    out = run(prices, models=["naive"])
    assert calls["n"] == 2 and "naive" in out["models"] and out["failed_models"] == {}


def test_model_registry_has_unique_names_and_known_families():
    names = [s[0] for s in pipeline.MODEL_SPECS]
    assert len(names) == len(set(names)) == 11
    assert {s[4] for s in pipeline.MODEL_SPECS} == {"baseline", "classical", "tree", "neural"}
