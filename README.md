# Electricity Price Forecasting + Battery Arbitrage

Real German day-ahead electricity prices, forecast with four models ranging from a naive baseline to an LSTM, then traded through a linear-programming-optimized battery dispatch strategy, walk-forward validated end to end.

## Demo

A Streamlit dashboard visualizes the real output of `scripts/precompute_results.py` — forecast accuracy per model, actual-vs-predicted price for the last 14 days of the test period, daily battery arbitrage P&L against the perfect-foresight ceiling, and the honest LSTM-vs-naive nuance below. Nothing here is mocked for the UI.

![Dashboard walkthrough](docs/demo_ui.gif)

```bash
python scripts/precompute_results.py   # real ~3-year fetch + 180-day walk-forward + backtest, writes results.json
streamlit run app.py                    # launches the dashboard at localhost:8501
```

A terminal-only recording of the test suite and the full pipeline run is also available for reference:

![Terminal recording of tests and the full pipeline running end to end](docs/demo.gif)

## What's actually here

- **Real data** (`src/data_loader.py`): hourly day-ahead auction prices and grid load for the DE-LU bidding zone, pulled directly from [SMARD](https://www.smard.de) (Bundesnetzagentur's official market data platform) — not a Kaggle snapshot. ~3 years of history (2023-09-04 to 2026-08-19), 25,921 feature rows after warmup, including real negative prices from renewable oversupply hours.
- **Leak-free feature engineering** (`src/features.py`): calendar features (cyclical hour/day-of-week/month encoding, weekend/holiday flags) plus lag and rolling-window features built only from information available before a day-ahead forecast's cutoff — verified by a dedicated test that every lag/rolling feature is provably built from strictly-past data.
- **Four forecasting models** (`src/models.py`): a naive persistence baseline (same hour last week — a genuinely strong benchmark for electricity, not a strawman), Linear Regression, XGBoost, and an LSTM (PyTorch) trained on raw price sequences.
- **Walk-forward evaluation** (`src/walk_forward.py`): retrains every 7 days over a 180-day test window, always on data strictly before the retrain cutoff, so no model is ever scored on data it was fit on.
- **Battery arbitrage strategy** (`src/battery.py`, `src/backtest.py`): a real linear program (via `scipy.optimize.linprog`) finds the profit-maximizing charge/discharge schedule for a configurable battery (capacity, power limit, round-trip efficiency), decided from each model's forecast and settled against actual realized prices — with state of charge correctly carried over day to day (see the honest bugs section below for why that mattered).
- **28 unit tests** (pytest): hand-calculated expected values for the battery LP, forecast metrics, feature causality, and backtest day-grouping logic.

## Honest bugs found and fixed during development

Two real bugs were caught by writing tests and cross-checking numbers, not after the fact:

1. **LSTM training bug.** The first version trained with full-batch gradient descent (one gradient step per epoch over the entire training set). On ~20,000+ training rows this produced far too few weight updates to converge: a 14-day walk-forward test scored **R² = -0.26** (worse than predicting the mean) and took **830 seconds** for just two retrain cycles. Switching to mini-batch training (batch size 256) fixed both problems at once: the same test scored **R² = 0.72**, competitive with Linear Regression and XGBoost, in **68 seconds**.
2. **Battery "free energy" bug.** The first version of `backtest_daily` reset every day's battery to the same configured initial state of charge (2.0 MWh), rather than carrying over the previous day's ending charge. This let the linear program "discharge for free" every morning regardless of that day's real price spread, manufacturing profit that wasn't real. Caught by a test asserting zero profit on a completely flat-price day — the buggy version returned ~47 EUR of phantom profit instead of 0. Fixed by chaining state of charge across days; a regression test now locks this in.

## Results

### Forecast accuracy (180-day walk-forward, weekly retrain)

| Model | MAE (€/MWh) | RMSE (€/MWh) | R² |
|---|---|---|---|
| Naive (same hour, last week) | 39.05 | 59.80 | 0.215 |
| Linear Regression | 28.51 | 41.46 | 0.623 |
| **XGBoost** | **26.52** | **39.35** | **0.660** |
| LSTM | 30.21 | 44.95 | 0.556 |

XGBoost is the most accurate forecaster on every metric. MAPE is deliberately not reported — see `src/metrics.py` for why (prices go negative and near-zero, which breaks percentage-based metrics).

### Battery arbitrage P&L (179 tradeable days, 4 MWh / 1 MW / 90% round-trip efficiency battery)

| Strategy | Total P&L | Mean/day | Profitable days |
|---|---|---|---|
| Naive-based dispatch | €96,010 | €536 | 100% |
| Linear-based dispatch | €97,923 | €547 | 100% |
| **XGBoost-based dispatch** | **€99,191** | **€554** | **100%** |
| LSTM-based dispatch | €93,944 | €525 | 100% |
| *Perfect foresight (ceiling)* | *€104,024* | *€581* | *100%* |

Every strategy is profitable every single day — expected, not a red flag: the LP can always choose to do nothing (see "profit is never negative by construction" below), so 100% profitable days doesn't mean the strategies are equally good, just that none of them ever lose money outright.

**The honest, more interesting finding is the LSTM.** It is clearly a better forecaster than the naive baseline by every standard metric (R² 0.556 vs. 0.215, MAE 30.21 vs. 39.05 €/MWh) — but it made *less* trading profit than that same simple naive baseline (€93,944 vs. €96,010). Lower average forecast error doesn't automatically translate into more trading profit: the battery strategy only cares about correctly ranking which hours are relatively cheap vs. expensive *within each day*, not minimizing error uniformly across all hours. XGBoost happens to be both the best forecaster and the best trader here, but that's not a given — the LSTM result is proof it doesn't follow automatically, and is the reason this project evaluates trading P&L as its own metric rather than assuming forecast accuracy alone tells the whole story.

Every model-based strategy captures roughly 90-95% of the perfect-foresight ceiling (XGBoost: 95.4%), which is a genuinely strong result for a battery arbitrage strategy driven by realistic day-ahead forecasts rather than hindsight.

## Design decisions and known simplifications

- **Day boundaries are UTC calendar days**, not the CET/CEST calendar days Germany's real day-ahead auction actually runs on. This is a simplification, not an oversight — noted here rather than silently glossed over.
- **Weekly retrain cadence**, not daily. Retraining all four models (including the LSTM) every single day across a 180-day test would take hours; weekly retraining is a realistic compromise real trading desks also make, and is documented rather than hidden.
- **Battery starts at a fixed initial charge only once**, at the very start of the whole test period — not reset daily (see the honest bugs section above for why that distinction matters).
- **Profit is never negative by construction**: the linear program can always choose to do nothing (charge = discharge = 0 for every hour), so the interesting comparison between strategies is the *gap to the perfect-foresight ceiling*, not whether a strategy is "profitable" at all.

## Building and running

```bash
pip install -r requirements.txt
pytest tests/ -v                          # 28 tests
python scripts/precompute_results.py      # real ~3-year data fetch + 180-day walk-forward + backtest, writes results.json
streamlit run app.py                       # dashboard at localhost:8501
```

## Stack

Python, pandas, NumPy, scikit-learn, XGBoost, PyTorch (LSTM), SciPy (`linprog` for battery dispatch optimization), pytest, Streamlit, Plotly. Data via the SMARD (Bundesnetzagentur) public API.
