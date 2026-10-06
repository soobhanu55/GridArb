# Cross-zone spread study (walk-forward)

Day-ahead prices of DE-LU, AT, FR, NL, CH, PL, CZ, daily means, 2023-10-15 to 2026-10-06 (1088 days). 21 candidate pairs; at a 5% Engle-Granger level about 1.1 pass by chance in any window. Paper P&L of 1 MW per leg (24 MWh per day), costs per MWh traded on both legs. Day-ahead zones are coupled by the auction, so this is a methodology study, not a tradable strategy.

| Cost (EUR/MWh) | Method | Total P&L (EUR) | Sharpe | Max drawdown (EUR) | Mean daily P&L, 95% CI (EUR) |
|---|---|---|---|---|---|
| 0.0 | naive (DE-LU-AT, look-ahead) | 48,451 | 4.25 | 573 | 46 (34 to 59) |
| 0.0 | walk-forward | 41,658 | 3.82 | 1,073 | 46 (32 to 59) |
| 0.5 | naive (DE-LU-AT, look-ahead) | 45,981 | 4.10 | 573 | 43 (33 to 56) |
| 0.5 | walk-forward | 39,676 | 3.70 | 1,090 | 44 (31 to 56) |
| 1.0 | naive (DE-LU-AT, look-ahead) | 43,511 | 3.93 | 573 | 41 (30 to 54) |
| 1.0 | walk-forward | 37,695 | 3.56 | 1,107 | 42 (29 to 54) |

Walk-forward: 30 windows (180 days formation, 30 days trading), 30 with a cointegrated pair. Pairs chosen: DE-LU-NL x6, AT-CH x5, DE-LU-PL x4, PL-CZ x4, NL-CZ x3, AT-FR x2, DE-LU-FR x2, AT-CZ x2, NL-PL x2. Windows with positive P&L at cost 0: 24 of 30.

**Do not read the Sharpe ratios as an edge.** Market coupling keeps neighbouring zones' prices tied together, so the screen found a pair in every window; the spreads are stationary, noisy series that always snap back; the strategy collects that reversion. No instrument lets you enter at one day's auction price and exit at the next day's, so this P&L cannot be realised; a real version needs a tradable spread product, margin and execution rules. Two things the study does show: walk-forward selection costs little against the look-ahead run here (the spread is stationary whichever pair is picked), and, unlike PairForge's original sector-ETF universe where nothing survived costs, a structurally coupled universe makes the screen find a pair in every window.
