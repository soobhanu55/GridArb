"""GridArb as a Prefect flow: the same steps as scripts/precompute_results.py, as tasks with retries, timeouts,
per-model results and a run history in the Prefect UI.

    pip install -r requirements-orchestration.txt
    python flows/gridarb_flow.py                      # run once (local ephemeral Prefect, no server or account)
    python flows/gridarb_flow.py --serve              # keep running; refreshes every Monday 06:00
    prefect server start                              # optional UI at http://127.0.0.1:4200

Why a flow: the data fetch hits a public API (retried with backoff), each model is an independent task (a failing
neural net does not discard finished tree models), and every run is recorded with its parameters and timings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from prefect import flow, task  # noqa: E402
from prefect.logging import get_run_logger  # noqa: E402

from src.battery import BatteryConfig  # noqa: E402
from src.data_loader import load_or_fetch  # noqa: E402
from src.features import build_feature_frame  # noqa: E402
from src import pipeline  # noqa: E402


@task(retries=3, retry_delay_seconds=[5, 30, 120])
def load_prices(n_weeks: int = 156, force_refresh: bool = False) -> pd.DataFrame:
    """Hourly DE-LU day-ahead prices and grid load from SMARD (cached on disk; the API is the flaky part)."""
    return load_or_fetch(n_weeks=n_weeks, force_refresh=force_refresh)


@task
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_feature_frame(df, wavelet=True)


@task(retries=1, retry_delay_seconds=10, timeout_seconds=3 * 3600)
def forecast(name: str, feat: pd.DataFrame, raw_price: pd.Series, test_days: int, retrain_every_days: int,
             cache_dir: str | None, fresh: bool):
    return pipeline.run_model(name, feat, raw_price, test_days, retrain_every_days,
                              Path(cache_dir) if cache_dir else None, fresh)


@task
def backtest(preds: pd.Series, feat: pd.DataFrame, cfg: BatteryConfig, perfect_total: float) -> dict:
    return pipeline.trade(preds, feat, cfg, perfect_total)


@flow(name="gridarb-pipeline", validate_parameters=False, log_prints=True)
def gridarb_pipeline(prices: pd.DataFrame | None = None, models: list[str] | None = None,
                     test_days: int = pipeline.TEST_DAYS, retrain_every_days: int = pipeline.RETRAIN_EVERY_DAYS,
                     fresh: bool = False, cache_dir: str | None = str(pipeline.CACHE_DIR),
                     out_path: str | None = str(ROOT / "results.json"), with_shap: bool = True,
                     track_mlflow: bool = True) -> dict:
    """Forecast with every model over a walk-forward test window, trade the forecasts through the battery LP,
    compare with perfect foresight, explain XGBoost with SHAP and write results.json. `prices` can be passed in
    (tests do); otherwise the data task fetches it."""
    log = get_run_logger()
    df = prices if prices is not None else load_prices()
    feat = make_features(df)
    raw_price = df["price_eur_mwh"]
    names = models or [spec[0] for spec in pipeline.MODEL_SPECS]
    cfg = BatteryConfig()
    results: dict = {"test_days": test_days, "retrain_every_days": retrain_every_days,
                     "battery_config": cfg.__dict__, "models": {}}

    run_ctx = None
    if track_mlflow:
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("gridarb-walk-forward")
        run_ctx = mlflow.start_run(run_name=f"prefect_walk_forward_{test_days}d")
        mlflow.log_params({"test_days": test_days, "retrain_every_days": retrain_every_days,
                           **{f"battery_{k}": v for k, v in cfg.__dict__.items()}})

    predictions: dict[str, pd.Series] = {}
    perfect_total = None
    failed: dict[str, str] = {}
    futures = {n: forecast.submit(n, feat, raw_price, test_days, retrain_every_days, cache_dir, fresh) for n in names}
    for name in names:
        try:
            preds, metrics, elapsed = futures[name].result()
        except Exception as exc:  # one failing model must not discard the others
            failed[name] = f"{type(exc).__name__}: {exc}"
            log.error("model %s failed: %s", name, failed[name])
            continue
        actual = feat.loc[preds.index, "price_eur_mwh"]
        if perfect_total is None:
            perfect_total = pipeline.perfect_foresight_total(actual, cfg)
        traded = backtest(preds, feat, cfg, perfect_total)
        predictions[name] = preds
        family = pipeline.SPEC_BY_NAME[name][4]
        results["models"][name] = {"family": family, "forecast_metrics": metrics, "fit_predict_seconds": elapsed, **traded}
        if track_mlflow:
            pipeline.log_model_run(name, preds, actual, metrics, traded, elapsed)
        log.info("%s: MAE=%.2f, P&L=%.0f EUR (%.1f%% of perfect foresight)", name, metrics["mae"],
                 traded["trading_summary"]["total_profit_eur"], traded["pct_of_perfect"])

    if not predictions:
        raise RuntimeError(f"every model failed: {failed}")
    pipeline.finalize(results, predictions, feat, cfg, png_path=ROOT / "docs" / "shap_summary.png" if with_shap else None,
                      with_shap=with_shap)
    results["failed_models"] = failed
    if out_path:
        Path(out_path).write_text(json.dumps(results, indent=2))
    if run_ctx is not None:
        if out_path:
            mlflow.log_artifact(out_path)
        mlflow.end_run()
    return results


if __name__ == "__main__":
    if "--serve" in sys.argv:
        gridarb_pipeline.serve(name="gridarb-weekly", cron="0 6 * * 1")
    else:
        out = gridarb_pipeline(fresh="--fresh" in sys.argv)
        print({k: round(v["forecast_metrics"]["mae"], 2) for k, v in out["models"].items()})
