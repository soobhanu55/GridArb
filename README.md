# GridArb — Electricity Price Forecasting + Battery Arbitrage

Real German day-ahead electricity prices, forecast with four models, then traded through a linear-programming battery dispatch strategy. Walk-forward validated end to end — nothing mocked.

![Dashboard walkthrough](docs/demo_ui.gif)
![Tests + full pipeline, terminal recording](docs/demo.gif)

```bash
python scripts/precompute_results.py   # ~3-year fetch + 180-day walk-forward + backtest
streamlit run app.py                    # dashboard at localhost:8501
```

## Results

**Forecast accuracy** (180-day walk-forward, weekly retrain):

| Model | MAE (€/MWh) | R² |
|---|---|---|
| Naive baseline | 39.05 | 0.215 |
| **XGBoost** | **26.52** | **0.660** |
| LSTM | 30.21 | 0.556 |

**Battery arbitrage P&L** (179 days, 4 MWh / 1 MW battery):

| Strategy | Total P&L | % of perfect-foresight ceiling |
|---|---|---|
| **XGBoost-based dispatch** | **€99,191** | **95.4%** |
| LSTM-based dispatch | €93,944 | 90.3% |
| *Perfect foresight* | *€104,024* | *100%* |

**The interesting finding:** LSTM is a clearly better forecaster than the naive baseline (R² 0.556 vs. 0.215) — but made *less* trading profit than that same naive baseline (€93,944 vs. €96,010). Lower average error doesn't automatically mean better trading decisions; the strategy only needs to correctly rank cheap vs. expensive hours *within each day*.

## Two real bugs, caught by testing

1. **LSTM training bug** — full-batch gradient descent converged so poorly it scored R² = -0.26 and took 830s for two retrain cycles. Mini-batch training fixed both: R² = 0.72, 68s.
2. **Battery "free energy" bug** — the backtest reset battery charge to a fixed value every day instead of carrying it over, letting the optimizer "discharge for free" each morning. Caught by a flat-price-day test expecting zero profit; the buggy version returned €47 of phantom profit.

## Stack

Python · pandas · scikit-learn · XGBoost · PyTorch (LSTM) · SciPy (`linprog`) · Streamlit · Plotly. Data via [SMARD](https://www.smard.de) (Bundesnetzagentur).

Full design decisions, known simplifications, and data details in [`docs/DETAILS.md`](docs/DETAILS.md).
