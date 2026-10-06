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

from src.battery import BatteryConfig
from src.data_loader import load_or_fetch
from src.features import build_feature_frame
from src.pipeline import (MODEL_SPECS, RETRAIN_EVERY_DAYS, ROOT, TEST_DAYS, finalize, log_model_run,
                          perfect_foresight_total, run_model, trade)

OUT_PATH = ROOT / "results.json"
SHAP_PNG = ROOT / "docs" / "shap_summary.png"


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
    for name, _factory, _uses_history, _cols, family in MODEL_SPECS:
        print(f"Walk-forward evaluating {name} (cached results are reused; --fresh recomputes)...")
        preds, metrics, elapsed = run_model(name, feat, raw_price, fresh="--fresh" in sys.argv)
        predictions[name] = preds
        print(f"  {name}: MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} R2={metrics['r2']:.3f} ({elapsed:.0f}s)")

        actual = feat.loc[preds.index, "price_eur_mwh"]
        if perfect_total is None:  # same test window for every model, so compute the ceiling once
            perfect_total = perfect_foresight_total(actual, battery_cfg)
        traded = trade(preds, feat, battery_cfg, perfect_total)
        bt_summary, pct_of_perfect = traded["trading_summary"], traded["pct_of_perfect"]
        print(f"  {name} trading: total={bt_summary['total_profit_eur']:.0f} EUR over {bt_summary['n_days']} days")

        log_model_run(name, preds, actual, metrics, traded, elapsed)

        results["models"][name] = {"family": family, "forecast_metrics": metrics, "fit_predict_seconds": elapsed, **traded}

    print("Computing perfect-foresight ceiling, chart tail and SHAP importances...")
    finalize(results, predictions, feat, battery_cfg, png_path=SHAP_PNG)
    print(f"  perfect foresight: total={results['perfect_foresight']['trading_summary']['total_profit_eur']:.0f} EUR")
    print("  top SHAP features:", ", ".join(results["shap"]["features"][:5]))

    OUT_PATH.write_text(json.dumps(results, indent=2))
    mlflow.log_artifact(str(OUT_PATH))
    mlflow.end_run()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
