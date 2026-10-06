"""Runs the full pipeline once: fetch data, forecast with every model over
a 180-day walk-forward test period, run the battery arbitrage backtest for
each model's forecasts plus a perfect-foresight upper bound, and cache
everything to results.json so the dashboard and demo scripts don't need to
re-run the (multi-minute) pipeline every time they're opened.

Every model run is also tracked in MLflow (params, metrics, predictions
artifact): `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
SHAP importances for the XGBoost forecaster are written to results.json
and docs/shap_summary.png.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import pandas as pd

from src.data_loader import load_or_fetch
from src.explain import global_importance, save_summary_plot, shap_values
from src.features import build_feature_frame, FEATURE_COLUMNS, FEATURE_COLUMNS_WAVELET
from src.walk_forward import walk_forward_evaluate
from src.models import (
    NaivePersistenceModel, LinearModel, XGBoostModel, RandomForestModel,
    HoltWintersModel, SARIMAModel, LSTMModel, MLPModel, CNNModel, CNNLSTMModel,
)
from src.backtest import backtest_daily, backtest_perfect_foresight, summarize_backtest
from src.battery import BatteryConfig

TEST_DAYS = 180
RETRAIN_EVERY_DAYS = 7
ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results.json"
SHAP_PNG = ROOT / "docs" / "shap_summary.png"
CACHE_DIR = ROOT / "data" / "cache"  # finished models are cached so an interrupted run can resume; --fresh ignores it

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


def _params(obj) -> dict:
    """Hyperparameters worth logging: the sklearn/XGBoost estimator's own, else plain attributes."""
    inner = getattr(obj, "model", None)
    raw = inner.get_params() if hasattr(inner, "get_params") else {
        k: v for k, v in vars(obj).items() if isinstance(v, (int, float, str))}
    return {k: str(v) for k, v in raw.items()}


def explain_xgboost(feat: pd.DataFrame, n_samples: int = 2000) -> dict:
    """SHAP on an XGBoost fitted only on pre-test data, explaining sampled test-period rows."""
    test_start = feat.index.max() - pd.Timedelta(days=TEST_DAYS)
    train, test = feat[feat.index < test_start], feat[feat.index >= test_start]
    model = XGBoostModel().fit(train[FEATURE_COLUMNS], train["price_eur_mwh"])
    X = test[FEATURE_COLUMNS].sample(min(n_samples, len(test)), random_state=42)
    values = shap_values(model.model, X)
    SHAP_PNG.parent.mkdir(exist_ok=True)
    save_summary_plot(values, X, str(SHAP_PNG))
    return {**global_importance(values, FEATURE_COLUMNS), "n_samples": len(X)}


def main() -> None:
    print("Loading data...")
    df = load_or_fetch()
    feat = build_feature_frame(df, wavelet=True)
    raw_price = df["price_eur_mwh"]
    print(f"  {len(feat)} feature rows, {feat.index.min()} to {feat.index.max()}")

    battery_cfg = BatteryConfig()
    results: dict = {"test_days": TEST_DAYS, "retrain_every_days": RETRAIN_EVERY_DAYS,
                      "battery_config": battery_cfg.__dict__, "models": {}}

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("gridarb-walk-forward")
    mlflow.start_run(run_name=f"walk_forward_{TEST_DAYS}d")
    mlflow.log_params({"test_days": TEST_DAYS, "retrain_every_days": RETRAIN_EVERY_DAYS,
                       **{f"battery_{k}": v for k, v in battery_cfg.__dict__.items()}})

    predictions: dict[str, pd.Series] = {}
    perfect_total = None
    for name, factory, uses_history, cols, family in MODEL_SPECS:
        cache_file = CACHE_DIR / f"{name}.pkl"
        if cache_file.exists() and "--fresh" not in sys.argv:
            preds, metrics, elapsed = pickle.loads(cache_file.read_bytes())
            print(f"Loaded cached {name} (originally {elapsed:.0f}s)")
        else:
            print(f"Walk-forward evaluating {name}...")
            t0 = time.time()
            preds, metrics = walk_forward_evaluate(
                feat, cols, "price_eur_mwh", factory,
                test_days=TEST_DAYS, retrain_every_days=RETRAIN_EVERY_DAYS,
                uses_price_history=uses_history, raw_price_series=raw_price if uses_history else None,
            )
            elapsed = time.time() - t0
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(pickle.dumps((preds, metrics, elapsed)))
        predictions[name] = preds
        print(f"  {name}: MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} R2={metrics['r2']:.3f} ({elapsed:.0f}s)")

        actual = feat.loc[preds.index, "price_eur_mwh"]
        bt = backtest_daily(preds, actual, battery_cfg)
        bt_summary = summarize_backtest(bt)
        print(f"  {name} trading: total={bt_summary['total_profit_eur']:.0f} EUR over {bt_summary['n_days']} days")

        if perfect_total is None:  # same test window for every model, so compute the ceiling once
            perfect_total = summarize_backtest(backtest_perfect_foresight(actual, battery_cfg))["total_profit_eur"]
        pct_of_perfect = 100 * bt_summary["total_profit_eur"] / perfect_total

        with mlflow.start_run(run_name=name, nested=True):
            mlflow.set_tags({"family": family, "uses_price_history": str(uses_history)})
            mlflow.log_params({"model_class": factory.__name__, "n_features": len(cols), **_params(factory())})
            mlflow.log_metrics({**metrics, "fit_predict_seconds": elapsed,
                                "total_profit_eur": bt_summary["total_profit_eur"],
                                "mean_daily_profit_eur": bt_summary["mean_daily_profit_eur"],
                                "pct_of_perfect_foresight": pct_of_perfect})
            pred_csv = ROOT / "data" / f"predictions_{name}.csv"
            pred_csv.parent.mkdir(exist_ok=True)
            pd.DataFrame({"predicted": preds, "actual": actual}).to_csv(pred_csv)
            mlflow.log_artifact(str(pred_csv))
            pred_csv.unlink()

        results["models"][name] = {
            "family": family,
            "forecast_metrics": metrics,
            "trading_summary": bt_summary,
            "pct_of_perfect": pct_of_perfect,
            "fit_predict_seconds": elapsed,
            "daily_profit": {str(k): v for k, v in bt["profit"].to_dict().items()},
        }

    print("Computing perfect-foresight upper bound...")
    any_preds = next(iter(predictions.values()))
    actual_test = feat.loc[any_preds.index, "price_eur_mwh"]
    bt_perfect = backtest_perfect_foresight(actual_test, battery_cfg)
    results["perfect_foresight"] = {
        "trading_summary": summarize_backtest(bt_perfect),
        "daily_profit": {str(k): v for k, v in bt_perfect["profit"].to_dict().items()},
    }
    print(f"  perfect foresight: total={results['perfect_foresight']['trading_summary']['total_profit_eur']:.0f} EUR")

    # Save a chunk of the actual vs predicted series (last 14 days) for charting.
    tail_index = any_preds.index[-24 * 14:]
    chart = {"timestamps": [t.isoformat() for t in tail_index],
             "actual": actual_test.loc[tail_index].tolist()}
    for name, preds in predictions.items():
        chart[f"pred_{name}"] = preds.loc[tail_index].tolist()
    results["chart_tail"] = chart

    print("Computing SHAP importances for XGBoost...")
    results["shap"] = explain_xgboost(feat)
    print("  top features:", ", ".join(results["shap"]["features"][:5]))

    OUT_PATH.write_text(json.dumps(results, indent=2))
    mlflow.log_artifact(str(OUT_PATH))
    mlflow.end_run()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
