# Does the zone graph help? (GNN study)

DE-LU day-ahead forecasts from the last 180 days, weekly retraining, 4301 hours. Inputs for the zone models: each of 7 zones' prices of the previous day and of the same day a week earlier (known at the day-ahead cutoff). 5 seeds per architecture; MAE/R2 are means over seeds with the standard deviation in brackets.

| Model | Sees | MAE EUR/MWh | R2 | P&L EUR | % of perfect foresight |
|---|---|---|---|---|---|
| local_mlp | DE-LU only | 26.40 (0.34) | 0.681 (0.005) | 108,010 | 95.9% |
| flat_mlp | all zones, no graph | 28.00 (0.69) | 0.652 (0.014) | 107,164 | 95.1% |
| gcn_no_edges | all zones, GCN without edges | 27.66 (0.26) | 0.662 (0.007) | 107,801 | 95.7% |
| gcn | all zones, GCN over borders | 27.18 (0.26) | 0.677 (0.004) | 108,015 | 95.9% |
| *naive (main benchmark)* | tabular features / price window | 41.07 | 0.248 | 106,156 | 94.3% |
| *linear (main benchmark)* | tabular features / price window | 30.65 | 0.631 | 106,540 | 94.6% |
| *random_forest (main benchmark)* | tabular features / price window | 28.10 | 0.657 | 107,093 | 95.1% |
| *xgboost (main benchmark)* | tabular features / price window | 29.69 | 0.646 | 107,040 | 95.0% |
| *transformer (main benchmark)* | tabular features / price window | 33.31 | 0.568 | 102,909 | 91.4% |

- gcn minus flat_mlp, daily MAE of the seed-averaged forecasts: -0.14 EUR/MWh (95% CI -1.35 to +1.04); negative favours gcn.
- gcn minus gcn_no_edges, daily MAE of the seed-averaged forecasts: -0.30 EUR/MWh (95% CI -1.27 to +0.69); negative favours gcn.
- gcn minus local_mlp, daily MAE of the seed-averaged forecasts: +0.52 EUR/MWh (95% CI -0.45 to +1.50); negative favours gcn.

**Reading.** The model that sees only DE-LU's own history (local_mlp) is the most accurate; giving the models the six neighbours' prices does not help, and the graph version is not reliably better than the same inputs without the graph (confidence intervals include zero). Neighbouring zones are coupled by the auction, so their prices carry little that DE-LU's own recent prices do not already contain. This is a negative result for this data, not evidence against graph networks in general.

**Comparison caveat.** The zone models see all 24 hours of the previous day, which are published before the day-ahead cutoff; the main benchmark's tabular features use only lags of 24 hours or more, a stricter rule. Part of the gap between these rows and the main benchmark rows therefore comes from the information set, not the architecture. Only comparisons within the zone-model rows are like for like. The reference rows are restricted to the same hours.
