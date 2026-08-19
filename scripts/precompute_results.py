"""Runs the full pipeline once: fetch data, forecast with all 4 models over
a 180-day walk-forward test period, run the battery arbitrage backtest for
each model's forecasts plus a perfect-foresight upper bound, and cache
everything to results.json so the dashboard and demo scripts don't need to
re-run the (multi-minute) pipeline every time they're opened.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data_loader import load_or_fetch
from src.features import build_feature_frame, FEATURE_COLUMNS
from src.walk_forward import walk_forward_evaluate
from src.models import NaivePersistenceModel, LinearModel, XGBoostModel, LSTMModel
from src.backtest import backtest_daily, backtest_perfect_foresight, summarize_backtest
from src.battery import BatteryConfig

TEST_DAYS = 180
RETRAIN_EVERY_DAYS = 7
OUT_PATH = Path(__file__).resolve().parent.parent / "results.json"


def main() -> None:
    print("Loading data...")
    df = load_or_fetch()
    feat = build_feature_frame(df)
    raw_price = df["price_eur_mwh"]
    print(f"  {len(feat)} feature rows, {feat.index.min()} to {feat.index.max()}")

    battery_cfg = BatteryConfig()
    results: dict = {"test_days": TEST_DAYS, "retrain_every_days": RETRAIN_EVERY_DAYS,
                      "battery_config": battery_cfg.__dict__, "models": {}}

    model_specs = [
        ("naive", NaivePersistenceModel, False),
        ("linear", LinearModel, False),
        ("xgboost", XGBoostModel, False),
        ("lstm", LSTMModel, True),
    ]

    predictions: dict[str, pd.Series] = {}
    for name, factory, uses_history in model_specs:
        print(f"Walk-forward evaluating {name}...")
        t0 = time.time()
        preds, metrics = walk_forward_evaluate(
            feat, FEATURE_COLUMNS, "price_eur_mwh", factory,
            test_days=TEST_DAYS, retrain_every_days=RETRAIN_EVERY_DAYS,
            uses_price_history=uses_history, raw_price_series=raw_price if uses_history else None,
        )
        elapsed = time.time() - t0
        predictions[name] = preds
        print(f"  {name}: MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} R2={metrics['r2']:.3f} ({elapsed:.0f}s)")

        actual = feat.loc[preds.index, "price_eur_mwh"]
        bt = backtest_daily(preds, actual, battery_cfg)
        bt_summary = summarize_backtest(bt)
        print(f"  {name} trading: total={bt_summary['total_profit_eur']:.0f} EUR over {bt_summary['n_days']} days")

        results["models"][name] = {
            "forecast_metrics": metrics,
            "trading_summary": bt_summary,
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

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
