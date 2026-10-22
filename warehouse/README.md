# GridArb warehouse (dbt + DuckDB)

The analytics layer over GridArb's results: the star-schema CSVs from `scripts/export_powerbi.py` are staged, modelled into marts and
tested with dbt on DuckDB (a local file, no server).

```bash
pip install dbt-duckdb
cd warehouse && dbt build --profiles-dir .          # 4 staging views, 3 marts, 26 tests
# your own run:  python scripts/export_powerbi.py && dbt build --profiles-dir . --vars '{data_dir: ../powerbi/data}'
```

| Layer | Models |
|---|---|
| staging | `stg_dim_model`, `stg_dim_date`, `stg_model_summary`, `stg_model_day`: typed and renamed, nothing else |
| marts | `mart_model_leaderboard` (one row per model: accuracy, profit, both rankings and their gap, days beating naive), `mart_daily_vs_naive` (every model-day against the naive baseline on the same day), `mart_model_month` |
| tests | unique / not-null / relationships / accepted values on every key, plus four SQL tests of the numbers themselves (below) |

## What the tests check, and one they taught me something with

* total profit over the window never exceeds perfect foresight; each model has a row for every test day; the daily profits add up to the
  per-model summary to within a euro; the ranking is consistent with the profits.
* **Daily profit never exceeding the perfect-foresight day** failed on 3 of 2,148 model-days (sarima on 2026-06-26 by 303 EUR,
  holt_winters twice). It is not a data error: the battery state of charge carries over from day to day, so each strategy has its own
  morning state of charge and can beat the perfect-foresight strategy on a single day. Only totals are bounded. That test is now a
  *warning* that documents this, and it means `pct_of_perfect_day` in the Power BI export can exceed 100% on a few days.

The Prefect flow can run this at the end of a run (`gridarb_pipeline(with_warehouse=True)`): it exports the star schema, runs `dbt build`
and fails the run if a test fails.
