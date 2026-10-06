"""The GridArb pipeline as plain functions, shared by scripts/precompute_results.py (a one-shot script) and
flows/gridarb_flow.py (the same steps as Prefect tasks with retries, caching and a run history)."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd

from src.backtest import backtest_daily, backtest_perfect_foresight, summarize_backtest
from src.battery import BatteryConfig
from src.explain import global_importance, save_summary_plot, shap_values
from src.features import FEATURE_COLUMNS, FEATURE_COLUMNS_WAVELET
from src.models import (CNNLSTMModel, CNNModel, HoltWintersModel, LinearModel, LSTMModel, MLPModel,
                        NaivePersistenceModel, RandomForestModel, SARIMAModel, XGBoostModel)
from src.walk_forward import walk_forward_evaluate

TEST_DAYS = 180
RETRAIN_EVERY_DAYS = 7
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"  # finished models are cached so an interrupted run can resume

# (key, factory, uses_price_history, feature columns, family)
MODEL_SPECS = [
    ("naive", NaivePersistenceModel, False, FEATURE_COLUMNS, "baseline"),
    ("linear", LinearModel, False, FEATURE_COLUMNS, "baseline"),
    ("holt_winters", HoltWintersModel, False, FEATURE_COLUMNS, "classical"),
    ("sarima", SARIMAModel, False, FEATURE_COLUMNS, "classical"),
    ("random_forest", RandomForestModel, False, FEATURE_COLUMNS, "tree"),
    ("xgboost", XGBoostModel, False, FEATURE_COLUMNS, "tree"),
    ("xgboost_wavelet", XGBoostModel, False, FEATURE_COLUMNS_WAVELET, "tree"),
    ("mlp", MLPModel, True, FEATURE_COLUMNS, "neural"),
    ("cnn", CNNModel, True, FEATURE_COLUMNS, "neural"),
    ("lstm", LSTMModel, True, FEATURE_COLUMNS, "neural"),
    ("cnn_lstm", CNNLSTMModel, True, FEATURE_COLUMNS, "neural"),
]
SPEC_BY_NAME = {s[0]: s for s in MODEL_SPECS}


def model_params(obj) -> dict:
    """Hyperparameters worth logging: the sklearn/XGBoost estimator's own, else plain attributes."""
    inner = getattr(obj, "model", None)
    raw = inner.get_params() if hasattr(inner, "get_params") else {
        k: v for k, v in vars(obj).items() if isinstance(v, (int, float, str))}
    return {k: str(v) for k, v in raw.items()}


def run_model(name: str, feat: pd.DataFrame, raw_price: pd.Series, test_days: int = TEST_DAYS,
              retrain_every_days: int = RETRAIN_EVERY_DAYS, cache_dir: Path | None = CACHE_DIR, fresh: bool = False):
    """Walk-forward forecast for one model. Returns (predictions, metrics, seconds). Finished runs are cached."""
    _, factory, uses_history, cols, _family = SPEC_BY_NAME[name]
    cache_file = cache_dir / f"{name}.pkl" if cache_dir else None
    if cache_file and cache_file.exists() and not fresh:
        return pickle.loads(cache_file.read_bytes())
    t0 = time.time()
    preds, metrics = walk_forward_evaluate(
        feat, cols, "price_eur_mwh", factory, test_days=test_days, retrain_every_days=retrain_every_days,
        uses_price_history=uses_history, raw_price_series=raw_price if uses_history else None)
    result = (preds, metrics, time.time() - t0)
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(pickle.dumps(result))
    return result


def perfect_foresight_total(actual: pd.Series, cfg: BatteryConfig) -> float:
    return summarize_backtest(backtest_perfect_foresight(actual, cfg))["total_profit_eur"]


def trade(preds: pd.Series, feat: pd.DataFrame, cfg: BatteryConfig, perfect_total: float) -> dict:
    """Battery backtest of one model's forecasts, settled at actual prices."""
    actual = feat.loc[preds.index, "price_eur_mwh"]
    bt = backtest_daily(preds, actual, cfg)
    summary = summarize_backtest(bt)
    return {"trading_summary": summary, "pct_of_perfect": 100 * summary["total_profit_eur"] / perfect_total,
            "daily_profit": {str(k): v for k, v in bt["profit"].to_dict().items()}}


def explain_xgboost(feat: pd.DataFrame, n_samples: int = 2000, png_path: Path | None = None) -> dict:
    """SHAP on an XGBoost fitted only on pre-test data, explaining sampled test-period rows."""
    test_start = feat.index.max() - pd.Timedelta(days=TEST_DAYS)
    train, test = feat[feat.index < test_start], feat[feat.index >= test_start]
    model = XGBoostModel().fit(train[FEATURE_COLUMNS], train["price_eur_mwh"])
    X = test[FEATURE_COLUMNS].sample(min(n_samples, len(test)), random_state=42)
    values = shap_values(model.model, X)
    if png_path is not None:
        png_path.parent.mkdir(exist_ok=True)
        save_summary_plot(values, X, str(png_path))
    return {**global_importance(values, FEATURE_COLUMNS), "n_samples": len(X)}


def log_model_run(name: str, preds: pd.Series, actual: pd.Series, metrics: dict, traded: dict, elapsed: float,
                  data_dir: Path = ROOT / "data") -> None:
    """One nested MLflow run per model: tags, params, forecast and trading metrics, predictions artifact.
    Call inside an active parent run."""
    import mlflow

    _, factory, uses_history, cols, family = SPEC_BY_NAME[name]
    summary = traded["trading_summary"]
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.set_tags({"family": family, "uses_price_history": str(uses_history)})
        mlflow.log_params({"model_class": factory.__name__, "n_features": len(cols), **model_params(factory())})
        mlflow.log_metrics({**metrics, "fit_predict_seconds": elapsed,
                            "total_profit_eur": summary["total_profit_eur"],
                            "mean_daily_profit_eur": summary["mean_daily_profit_eur"],
                            "pct_of_perfect_foresight": traded["pct_of_perfect"]})
        pred_csv = data_dir / f"predictions_{name}.csv"
        pred_csv.parent.mkdir(exist_ok=True)
        pd.DataFrame({"predicted": preds, "actual": actual}).to_csv(pred_csv)
        mlflow.log_artifact(str(pred_csv))
        pred_csv.unlink()


def finalize(results: dict, predictions: dict[str, pd.Series], feat: pd.DataFrame, cfg: BatteryConfig,
             png_path: Path | None = None, with_shap: bool = True) -> dict:
    """Adds the perfect-foresight ceiling, the chart tail (last 14 days of actual vs predicted) and the SHAP block."""
    any_preds = next(iter(predictions.values()))
    actual_test = feat.loc[any_preds.index, "price_eur_mwh"]
    bt_perfect = backtest_perfect_foresight(actual_test, cfg)
    results["perfect_foresight"] = {
        "trading_summary": summarize_backtest(bt_perfect),
        "daily_profit": {str(k): v for k, v in bt_perfect["profit"].to_dict().items()},
    }
    tail_index = any_preds.index[-24 * 14:]
    chart = {"timestamps": [t.isoformat() for t in tail_index], "actual": actual_test.loc[tail_index].tolist()}
    for name, preds in predictions.items():
        chart[f"pred_{name}"] = preds.loc[tail_index].tolist()
    results["chart_tail"] = chart
    if with_shap:
        results["shap"] = explain_xgboost(feat, png_path=png_path)
    return results
