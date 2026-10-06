"""Streamlit dashboard for the electricity price forecasting + battery
arbitrage project. Reads the real results.json written by
scripts/precompute_results.py -- no numbers here are mocked for the UI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.json"

MODEL_LABELS = {
    "naive": "Naive (persistence)", "linear": "Linear Regression",
    "holt_winters": "Holt-Winters", "sarima": "SARIMA",
    "random_forest": "Random Forest", "xgboost": "XGBoost", "xgboost_wavelet": "XGBoost + wavelet feature",
    "mlp": "MLP", "cnn": "1D-CNN", "lstm": "LSTM", "cnn_lstm": "CNN-LSTM",
}
MODEL_COLORS = {
    "naive": "#8b949e", "linear": "#58a6ff", "holt_winters": "#79c0ff", "sarima": "#a5d6ff",
    "random_forest": "#56d364", "xgboost": "#3fb950", "xgboost_wavelet": "#238636",
    "mlp": "#ffa657", "cnn": "#f0883e", "lstm": "#e3b341", "cnn_lstm": "#d29922",
}

st.set_page_config(page_title="Electricity Price Forecasting + Battery Arbitrage", layout="wide", page_icon="⚡")

st.markdown(
    """
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #010409; }
    div[data-testid="stMetric"] {
        background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] { color: #8b949e; }
    h1, h2, h3 { color: #e6edf3; }
    .src-table { width: 100%; border-collapse: collapse; font-family: Consolas, monospace; font-size: 13px; }
    .src-table th { text-align: left; color: #8b949e; border-bottom: 1px solid #30363d; padding: 6px 10px; }
    .src-table td { color: #c9d1d9; border-bottom: 1px solid #21262d; padding: 6px 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ Electricity Price Forecasting + Battery Arbitrage")
st.caption(
    "Real German day-ahead prices (SMARD/Bundesnetzagentur) forecast with 11 models, then traded through an "
    "LP-optimized battery dispatch, walk-forward validated. All numbers below come from a real run of "
    "`scripts/precompute_results.py`, tracked in MLflow."
)

if not RESULTS_PATH.exists():
    st.error(f"{RESULTS_PATH.name} not found. Run `python scripts/precompute_results.py` first.")
    st.stop()

results = json.loads(RESULTS_PATH.read_text())
models = {k: v for k, v in results["models"].items() if k in MODEL_LABELS}
perfect = results["perfect_foresight"]

st.markdown("---")
st.subheader(f"Model comparison ({results['test_days']}-day walk-forward test, weekly retrain)")
table = pd.DataFrame([
    {
        "Model": MODEL_LABELS[k], "Family": m["family"],
        "MAE (€/MWh)": m["forecast_metrics"]["mae"], "RMSE": m["forecast_metrics"]["rmse"],
        "R²": m["forecast_metrics"]["r2"],
        "Total P&L (€)": m["trading_summary"]["total_profit_eur"],
        "% of perfect foresight": m["pct_of_perfect"],
    }
    for k, m in models.items()
]).sort_values("MAE (€/MWh)")
st.dataframe(
    table, hide_index=True, use_container_width=True, height=36 * (len(table) + 1) + 3,
    column_config={
        "MAE (€/MWh)": st.column_config.NumberColumn(format="%.2f"),
        "RMSE": st.column_config.NumberColumn(format="%.2f"),
        "R²": st.column_config.NumberColumn(format="%.3f"),
        "Total P&L (€)": st.column_config.NumberColumn(format="%.0f"),
        "% of perfect foresight": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

chosen = st.multiselect(
    "Models to plot", list(models), default=[k for k in ("naive", "xgboost", "lstm") if k in models],
    format_func=MODEL_LABELS.get,
)

st.markdown("---")
left, right = st.columns([2, 1])

with left:
    st.subheader("Actual vs. forecast price (last 14 days of test period)")
    chart = results["chart_tail"]
    ts = pd.to_datetime(chart["timestamps"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=chart["actual"], name="Actual", line=dict(color="#f0f6fc", width=2)))
    for key in chosen:
        fig.add_trace(go.Scatter(x=ts, y=chart[f"pred_{key}"], name=MODEL_LABELS[key],
                                  line=dict(color=MODEL_COLORS[key], width=1), opacity=0.75))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="EUR/MWh",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Battery arbitrage: daily profit by model")
    fig2 = go.Figure()
    for key in chosen:
        daily = models[key]["daily_profit"]
        dates = pd.to_datetime(list(daily.keys()))
        fig2.add_trace(go.Scatter(x=dates, y=list(daily.values()), name=MODEL_LABELS[key],
                                   line=dict(color=MODEL_COLORS[key], width=1.2)))
    perfect_daily = perfect["daily_profit"]
    fig2.add_trace(go.Scatter(x=pd.to_datetime(list(perfect_daily.keys())), y=list(perfect_daily.values()),
                               name="Perfect foresight (ceiling)", line=dict(color="#a371f7", width=1.5, dash="dot")))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="EUR / day",
    )
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.subheader("Trading P&L summary")
    rows_html = ""
    for key in sorted(models, key=lambda k: -models[k]["trading_summary"]["total_profit_eur"]):
        s = models[key]["trading_summary"]
        rows_html += (f"<tr><td>{MODEL_LABELS[key]}</td><td>{s['total_profit_eur']:,.0f} €</td>"
                       f"<td>{s['mean_daily_profit_eur']:.1f} €</td>"
                       f"<td>{s['pct_profitable_days']:.0f}%</td></tr>")
    ps = perfect["trading_summary"]
    rows_html += (f"<tr><td><i>Perfect foresight</i></td><td><i>{ps['total_profit_eur']:,.0f} €</i></td>"
                   f"<td><i>{ps['mean_daily_profit_eur']:.1f} €</i></td>"
                   f"<td><i>{ps['pct_profitable_days']:.0f}%</i></td></tr>")
    st.markdown(
        f"""
        <table class="src-table">
        <tr><th>Strategy</th><th>Total P&L</th><th>Mean/day</th><th>Profitable days</th></tr>
        {rows_html}
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Battery config")
    st.json(results["battery_config"])

    st.subheader("Test setup")
    st.markdown(
        f"- **Test period:** {results['test_days']} days\n"
        f"- **Retrain cadence:** every {results['retrain_every_days']} days\n"
        f"- Data source: SMARD (Bundesnetzagentur), real DE-LU day-ahead prices"
    )

st.markdown("---")
st.subheader("What drives the XGBoost forecast? (SHAP)")
shap_res = results.get("shap")
if shap_res:
    top = list(zip(shap_res["features"], shap_res["mean_abs_shap"]))[:10][::-1]
    fig3 = go.Figure(go.Bar(x=[v for _, v in top], y=[f for f, _ in top], orientation="h", marker_color="#3fb950"))
    fig3.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="mean |SHAP| (€/MWh moved per prediction)",
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(f"Computed on {shap_res['n_samples']} sampled test-period rows by an XGBoost fit only on pre-test data.")
else:
    st.info("Re-run `scripts/precompute_results.py` to compute SHAP importances.")

st.markdown("---")
st.subheader("Best forecaster → best trader?")
best_forecast = min(models, key=lambda k: models[k]["forecast_metrics"]["mae"])
best_trader = max(models, key=lambda k: models[k]["trading_summary"]["total_profit_eur"])
st.write(
    f"Lowest forecast error: **{MODEL_LABELS[best_forecast]}** "
    f"(MAE {models[best_forecast]['forecast_metrics']['mae']:.1f} €/MWh). "
    f"Highest trading profit: **{MODEL_LABELS[best_trader]}** "
    f"({models[best_trader]['trading_summary']['total_profit_eur']:,.0f} € total). "
    + ("These agree." if best_forecast == best_trader else
       "These are **different models** — a lower MAE doesn't automatically mean more trading profit, "
       "since the battery strategy only cares about getting the *shape* of price spikes/dips right at the "
       "right hours, not minimizing average error across all hours equally.")
)

naive = models["naive"]
naive_r2, naive_pnl = naive["forecast_metrics"]["r2"], naive["trading_summary"]["total_profit_eur"]
paradox = [k for k, m in models.items()
           if m["forecast_metrics"]["r2"] > naive_r2 and m["trading_summary"]["total_profit_eur"] < naive_pnl]
if paradox:
    names = ", ".join(MODEL_LABELS[k] for k in paradox)
    st.info(
        f"**Accuracy ≠ trading value.** {names} beat the naive baseline on R² "
        f"(naive {naive_r2:.3f}) yet made **less** trading profit than it "
        f"({naive_pnl:,.0f} € total). That is why this project evaluates both."
    )
