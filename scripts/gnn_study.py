"""Does the interconnection graph help? Walk-forward comparison of zone-level models (writes docs/gnn_study.md).

    python scripts/gnn_study.py

Same 180-day test window, weekly retraining, metrics and battery backtest as the main benchmark. Each architecture is
trained with 5 seeds; the table reports the mean and spread over seeds, and a paired block bootstrap over days compares
the graph model with the models that see the same inputs without the graph.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import pipeline  # noqa: E402
from src.battery import BatteryConfig  # noqa: E402
from src.data_loader import load_or_fetch, load_zone_prices  # noqa: E402
from src.features import build_feature_frame  # noqa: E402
from src.metrics import summarize  # noqa: E402
from src.zone_gnn import ARCHS, walk_forward_zone  # noqa: E402

SEEDS = range(5)


def paired_day_bootstrap(err_a: pd.Series, err_b: pd.Series, n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean difference in daily MAE (a - b) with a 95% bootstrap interval over days."""
    d = (err_a.groupby(err_a.index.date).mean() - err_b.groupby(err_b.index.date).mean()).to_numpy()
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    hourly = load_zone_prices()
    df = load_or_fetch()
    feat = build_feature_frame(df, wavelet=True)
    test_start, end = feat.index.max() - pd.Timedelta(days=pipeline.TEST_DAYS), feat.index.max()
    cfg = BatteryConfig()
    actual = feat.loc[feat.index >= test_start, "price_eur_mwh"]

    preds_by_arch: dict[str, list[pd.Series]] = {}
    for arch in ARCHS:
        preds_by_arch[arch] = [walk_forward_zone(hourly, arch, test_start, end, seed=s) for s in SEEDS]
        print(arch, "done", flush=True)

    idx = preds_by_arch[ARCHS[0]][0].index
    act = feat.loc[idx, "price_eur_mwh"]
    perfect = pipeline.perfect_foresight_total(act, cfg)

    lines = ["# Does the zone graph help? (GNN study)\n",
             f"DE-LU day-ahead forecasts from the last 180 days, weekly retraining, {len(idx)} hours. Inputs for the zone models: "
             "each of 7 zones' prices of the previous day and of the same day a week earlier (known at the day-ahead cutoff). "
             "5 seeds per architecture; MAE/R2 are means over seeds with the standard deviation in brackets.\n",
             "| Model | Sees | MAE EUR/MWh | R2 | P&L EUR | % of perfect foresight |", "|---|---|---|---|---|---|"]
    sees = {"local_mlp": "DE-LU only", "flat_mlp": "all zones, no graph", "gcn_no_edges": "all zones, GCN without edges",
            "gcn": "all zones, GCN over borders"}
    mean_pred = {}
    for arch in ARCHS:
        runs = [p.loc[idx] for p in preds_by_arch[arch]]
        m = [summarize(act.to_numpy(), p.to_numpy()) for p in runs]
        pnl = [pipeline.trade(p, feat, cfg, perfect)["trading_summary"]["total_profit_eur"] for p in runs]
        mean_pred[arch] = sum(runs) / len(runs)
        lines.append(f"| {arch} | {sees[arch]} | {np.mean([x['mae'] for x in m]):.2f} ({np.std([x['mae'] for x in m]):.2f}) | "
                     f"{np.mean([x['r2'] for x in m]):.3f} ({np.std([x['r2'] for x in m]):.3f}) | {np.mean(pnl):,.0f} | {100 * np.mean(pnl) / perfect:.1f}% |")
    for name in ("naive", "linear", "random_forest", "xgboost", "transformer"):  # main benchmark, restricted to the same hours
        cached = pipeline.CACHE_DIR / f"{name}.pkl"
        if cached.exists():
            ref = pickle.loads(cached.read_bytes())[0].loc[idx]
            m = summarize(act.to_numpy(), ref.to_numpy())
            t = pipeline.trade(ref, feat, cfg, perfect)
            lines.append(f"| *{name} (main benchmark)* | tabular features / price window | {m['mae']:.2f} | {m['r2']:.3f} | "
                         f"{t['trading_summary']['total_profit_eur']:,.0f} | {t['pct_of_perfect']:.1f}% |")
    lines.append("")
    for a, b in (("gcn", "flat_mlp"), ("gcn", "gcn_no_edges"), ("gcn", "local_mlp")):
        ea, eb = (mean_pred[a] - act).abs(), (mean_pred[b] - act).abs()
        d, lo, hi = paired_day_bootstrap(ea, eb)
        lines.append(f"- {a} minus {b}, daily MAE of the seed-averaged forecasts: {d:+.2f} EUR/MWh (95% CI {lo:+.2f} to {hi:+.2f}); negative favours {a}.")
    lines += ['', "**Reading.** The model that sees only DE-LU's own history (local_mlp) is the most accurate; giving the models the six neighbours' prices does not help, and the graph version is not reliably better than the same inputs without the graph (confidence intervals include zero). Neighbouring zones are coupled by the auction, so their prices carry little that DE-LU's own recent prices do not already contain. This is a negative result for this data, not evidence against graph networks in general.", '', "**Comparison caveat.** The zone models see all 24 hours of the previous day, which are published before the day-ahead cutoff; the main benchmark's tabular features use only lags of 24 hours or more, a stricter rule. Part of the gap between these rows and the main benchmark rows therefore comes from the information set, not the architecture. Only comparisons within the zone-model rows are like for like. The reference rows are restricted to the same hours."]
    out = ROOT / "docs" / "gnn_study.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
