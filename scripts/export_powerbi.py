"""Exports the forecasting and battery-trading results as a star schema for Power BI (one CSV per table).

    python scripts/export_powerbi.py            # reads results.json (+ data/cache/*.pkl for per-day errors), writes powerbi/data

Tables: dim_model, dim_date, fact_model_summary (one row per model), fact_model_day (error and profit per model per day),
fact_hourly_tail (actual vs predicted for the last 14 days, every model). Relationships and measures: powerbi/README.md.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def build_tables(results: dict, preds: dict[str, pd.Series] | None = None, actual: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    names = list(results["models"])
    dim_model = pd.DataFrame({
        "model_key": range(1, len(names) + 1), "model": names,
        "family": [results["models"][n]["family"] for n in names],
        "is_neural": [int(results["models"][n]["family"] == "neural") for n in names]})
    key = dict(zip(dim_model["model"], dim_model["model_key"]))

    summary = pd.DataFrame([{
        "model_key": key[n], "mae": m["forecast_metrics"]["mae"], "rmse": m["forecast_metrics"]["rmse"], "r2": m["forecast_metrics"]["r2"],
        "total_profit_eur": m["trading_summary"]["total_profit_eur"], "pct_of_perfect": m["pct_of_perfect"],
        "fit_predict_seconds": m["fit_predict_seconds"]} for n, m in results["models"].items()])
    perfect = results["perfect_foresight"]["trading_summary"]["total_profit_eur"]
    summary["perfect_profit_eur"] = perfect

    perfect_daily = {str(k)[:10]: v for k, v in results["perfect_foresight"]["daily_profit"].items()}
    rows = []
    for n, m in results["models"].items():
        for day, profit in m["daily_profit"].items():
            d = str(day)[:10]
            rows.append({"date_key": int(d.replace("-", "")), "model_key": key[n], "profit_eur": profit,
                         "perfect_profit_eur": perfect_daily.get(d, np.nan)})
    fact_day = pd.DataFrame(rows)
    if preds and actual is not None:
        errs = []
        for n, p in preds.items():
            a = actual.loc[p.index]
            e = pd.DataFrame({"err": (p - a).to_numpy(), "abs": (p - a).abs().to_numpy()}, index=p.index)
            g = e.groupby(e.index.strftime("%Y%m%d").astype(int)).agg(mae=("abs", "mean"), bias=("err", "mean"))
            g["model_key"] = key[n]
            errs.append(g.rename_axis("date_key").reset_index())
        fact_day = fact_day.merge(pd.concat(errs), on=["date_key", "model_key"], how="left")
    fact_day["pct_of_perfect_day"] = np.where(fact_day["perfect_profit_eur"] > 0, fact_day["profit_eur"] / fact_day["perfect_profit_eur"], np.nan)

    dates = pd.to_datetime(fact_day["date_key"].astype(str), format="%Y%m%d").drop_duplicates().sort_values()
    dim_date = pd.DataFrame({"date_key": dates.dt.strftime("%Y%m%d").astype(int).to_numpy(), "date": dates.dt.strftime("%Y-%m-%d").to_numpy(),
                             "year": dates.dt.year.to_numpy(), "month": dates.dt.month.to_numpy(), "month_name": dates.dt.strftime("%B").to_numpy(),
                             "weekday": dates.dt.strftime("%A").to_numpy(), "weekday_number": (dates.dt.weekday + 1).to_numpy(),
                             "is_weekend": (dates.dt.weekday >= 5).astype(int).to_numpy()})

    tail = results["chart_tail"]
    ts = pd.to_datetime(tail["timestamps"], utc=True)
    hourly = pd.concat([pd.DataFrame({"timestamp": ts, "model_key": key[n], "predicted": tail[f"pred_{n}"], "actual": tail["actual"]})
                        for n in names if f"pred_{n}" in tail], ignore_index=True)
    return {"dim_model": dim_model, "dim_date": dim_date, "fact_model_summary": summary, "fact_model_day": fact_day, "fact_hourly_tail": hourly}


def export(out_dir: Path | str = ROOT / "powerbi" / "data", results_path: Path = ROOT / "results.json", cache_dir: Path = ROOT / "data" / "cache") -> dict[str, int]:
    results = json.loads(Path(results_path).read_text())
    preds, actual = {}, None
    for n in results["models"]:
        f = Path(cache_dir) / f"{n}.pkl"
        if f.exists():
            preds[n] = pickle.loads(f.read_bytes())[0]
    if preds:
        raw = pd.read_csv(ROOT / "data" / "electricity_raw.csv", index_col=0, parse_dates=True)["price_eur_mwh"]
        actual = raw
    tables = build_tables(results, preds or None, actual)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    return {n: len(d) for n, d in tables.items()}


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    for table, n in export().items():
        print(f"{table}: {n} rows")
