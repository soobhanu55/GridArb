# GridArb — Full Details

Real German day-ahead electricity prices, forecast with eleven models (naive baseline, classical, tree-based and neural), then traded through a linear-programming-optimized battery dispatch strategy, walk-forward validated end to end.

## What's actually here

- **Real data** (`src/data_loader.py`): hourly day-ahead auction prices and grid load for the DE-LU bidding zone, pulled directly from [SMARD](https://www.smard.de) (Bundesnetzagentur's official market data platform) — not a Kaggle snapshot. ~3 years of history (2023-10 to 2026-10), 25,895 feature rows after warmup, including real negative prices from renewable oversupply hours.
- **Leak-free feature engineering** (`src/features.py`): calendar features (cyclical hour/day-of-week/month encoding, weekend/holiday flags) plus lag and rolling-window features built only from information available before a day-ahead forecast's cutoff — verified by a dedicated test that every lag/rolling feature is provably built from strictly-past data.
- **Eleven forecasting models** (`src/models.py`): naive persistence (same hour last week, a genuinely strong benchmark), Linear Regression, Holt-Winters and SARIMA (classical), Random Forest and XGBoost (tree), and four PyTorch networks on raw price windows (MLP, 1D-CNN, LSTM, CNN-LSTM) that share one `SequenceModel` base. The neural families and the Random Forest / Holt-Winters / wavelet ideas come from two earlier notebook projects (ForecastNet, DemandLens), re-implemented here on real electricity data inside the same leak-free harness.
- **Causal wavelet feature** (`src/features.py`): a wavelet-denoised price at t-24h, computed from a trailing window only (denoising the whole series would leak the future; a test asserts it does not). Included as an ablation against plain XGBoost.
- **Experiment tracking** (`scripts/precompute_results.py`): one MLflow parent run, one child run per model, with params, MAE/RMSE/R², P&L, % of ceiling, runtime and a predictions artifact.
- **Explainability** (`src/explain.py`): SHAP values for the XGBoost forecaster; `price_lag_24h` dominates (mean |SHAP| 34.0 €/MWh, ~3.5x the next feature).
- **Walk-forward evaluation** (`src/walk_forward.py`): retrains every 7 days over a 180-day test window, always on data strictly before the retrain cutoff, so no model is ever scored on data it was fit on.
- **Battery arbitrage strategy** (`src/battery.py`, `src/backtest.py`): a real linear program (via `scipy.optimize.linprog`) finds the profit-maximizing charge/discharge schedule for a configurable battery (capacity, power limit, round-trip efficiency), decided from each model's forecast and settled against actual realized prices — with state of charge correctly carried over day to day (see the honest bugs section below for why that mattered).
- **42 unit tests** (pytest, run in GitHub Actions CI): hand-calculated expected values for the battery LP, forecast metrics, feature and wavelet causality, backtest day-grouping, model output shapes, seeded reproducibility and SHAP additivity.

## Honest bugs found and fixed during development

Two real bugs were caught by writing tests and cross-checking numbers, not after the fact:

1. **LSTM training bug.** The first version trained with full-batch gradient descent (one gradient step per epoch over the entire training set). On ~20,000+ training rows this produced far too few weight updates to converge: a 14-day walk-forward test scored **R² = -0.26** (worse than predicting the mean) and took **830 seconds** for just two retrain cycles. Switching to mini-batch training (batch size 256) fixed both problems at once: the same test scored **R² = 0.72**, competitive with Linear Regression and XGBoost, in **68 seconds**.
2. **Battery "free energy" bug.** The first version of `backtest_daily` reset every day's battery to the same configured initial state of charge (2.0 MWh), rather than carrying over the previous day's ending charge. This let the linear program "discharge for free" every morning regardless of that day's real price spread, manufacturing profit that wasn't real. Caught by a test asserting zero profit on a completely flat-price day — the buggy version returned ~47 EUR of phantom profit instead of 0. Fixed by chaining state of charge across days; a regression test now locks this in.

## Results

Numbers below come from one real run of `scripts/precompute_results.py` (also stored in MLflow and `results.json`).

### Forecast accuracy (180-day walk-forward, weekly retrain)

| Model | Family | MAE (€/MWh) | RMSE (€/MWh) | R² |
|---|---|---|---|---|
| Random Forest | tree | 28.23 | 43.78 | 0.656 |
| XGBoost | tree | 29.87 | 44.54 | 0.644 |
| XGBoost + wavelet feature | tree | 29.94 | 44.57 | 0.644 |
| Linear Regression | baseline | 30.74 | 45.35 | 0.631 |
| LSTM | neural | 32.96 | 50.03 | 0.551 |
| CNN-LSTM | neural | 33.33 | 49.89 | 0.554 |
| MLP | neural | 35.06 | 50.96 | 0.534 |
| SARIMA | classical | 40.36 | 57.58 | 0.406 |
| Naive (same hour, last week) | baseline | 41.07 | 64.56 | 0.253 |
| 1D-CNN | neural | 47.58 | 63.69 | 0.273 |
| Holt-Winters | classical | 66.38 | 87.71 | -0.379 |

MAPE is deliberately not reported: prices go negative and near-zero, which breaks percentage metrics (see `src/metrics.py`).

### Battery arbitrage P&L (179 tradeable days, 4 MWh / 1 MW / 90% round-trip efficiency battery)

| Strategy | Total P&L | Mean/day | Profitable days | % of ceiling |
|---|---|---|---|---|
| Random Forest-based dispatch | €107,093 | €598 | 100% | 95.1% |
| XGBoost-based dispatch | €107,040 | €598 | 100% | 95.0% |
| XGBoost + wavelet feature-based dispatch | €106,980 | €598 | 100% | 95.0% |
| Linear Regression-based dispatch | €106,540 | €595 | 100% | 94.6% |
| Naive (same hour, last week)-based dispatch | €106,156 | €593 | 100% | 94.3% |
| Holt-Winters-based dispatch | €104,792 | €585 | 100% | 93.0% |
| MLP-based dispatch | €103,726 | €579 | 100% | 92.1% |
| SARIMA-based dispatch | €102,756 | €574 | 100% | 91.2% |
| LSTM-based dispatch | €101,565 | €567 | 100% | 90.2% |
| CNN-LSTM-based dispatch | €101,401 | €566 | 100% | 90.0% |
| 1D-CNN-based dispatch | €97,782 | €546 | 100% | 86.8% |
| *Perfect foresight (ceiling)* | *€112,628* | *€629* | *100%* | *100%* |

Every strategy is profitable every day by construction: the LP can always choose to do nothing, so profitability alone says nothing. The informative comparison is the gap to the perfect-foresight ceiling.

### Findings

- **Tree models beat deep nets** on both error and profit. The four neural nets all read the same raw price window under the same no-leakage rule and land at R² 0.27 to 0.55; Random Forest and XGBoost reach 0.64 to 0.66. Hyperparameters are defaults, not tuned, so this is a result for these settings.
- **Accuracy and trading value diverge.** SARIMA, MLP, CNN, LSTM and CNN-LSTM beat the naive baseline on R² but earned less than it. The battery strategy only needs the cheap vs. expensive hours ranked correctly within each day.
- **The upside over naive is small:** the best model earns about €940 more than "same hour last week" over 179 days (under 1%). Even Holt-Winters (R² -0.38) captures 93% of the ceiling, because the daily price shape alone carries most of the arbitrage value.
- **Wavelet feature: null result.** MAE 29.94 vs 29.87, P&L within €60 of plain XGBoost.
- **GPU effect:** the LSTM walk-forward took 1,822 s on CPU and 71 s on the RTX 4050 with essentially the same scores (R² 0.556 vs 0.551), so GPU training is a speed change, not an accuracy change.

## Design decisions and known simplifications

- **Day boundaries are UTC calendar days**, not the CET/CEST calendar days Germany's real day-ahead auction actually runs on. This is a simplification, not an oversight — noted here rather than silently glossed over.
- **Weekly retrain cadence**, not daily. Retraining all eleven models every single day across a 180-day test would take hours; weekly retraining is a realistic compromise real trading desks also make, and is documented rather than hidden.
- **Battery starts at a fixed initial charge only once**, at the very start of the whole test period — not reset daily (see the honest bugs section above for why that distinction matters).
- **Profit is never negative by construction**: the linear program can always choose to do nothing (charge = discharge = 0 for every hour), so the interesting comparison between strategies is the *gap to the perfect-foresight ceiling*, not whether a strategy is "profitable" at all.

## Building and running

```bash
pip install -r requirements.txt
pytest tests/ -v                          # 42 tests
python scripts/precompute_results.py      # data fetch + 11-model walk-forward + backtest + MLflow + SHAP, writes results.json
python scripts/precompute_results.py --fresh   # ignore the per-model cache in data/cache/
mlflow ui --backend-store-uri sqlite:///mlflow.db
streamlit run app.py                       # dashboard at localhost:8501
```

## Stack

Python, pandas, NumPy, scikit-learn, XGBoost, PyTorch, statsmodels, PyWavelets, SHAP, MLflow, SciPy (`linprog` for battery dispatch optimization), pytest, GitHub Actions, Streamlit, Plotly. Data via the SMARD (Bundesnetzagentur) public API.
