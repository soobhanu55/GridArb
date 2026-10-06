# Power BI report: GridArb model comparison

I cannot produce a `.pbix` file (Power BI Desktop is a Windows GUI application), so this folder holds everything that goes into one: the data model as
CSV files exported from the real results, the DAX measures, and a page-by-page build guide. Building it takes about 30 minutes in Power BI Desktop (free).

```bash
python scripts/export_powerbi.py      # results.json + cached forecasts -> powerbi/data/*.csv (committed sample from the real run)
```

## Data model (star schema)

| Table | Grain | Key columns |
|---|---|---|
| `dim_model` | one row per forecasting model | `model_key`, `model`, `family`, `is_neural` |
| `dim_date` | one row per test day (179) | `date_key` (yyyymmdd), `date`, `month_name`, `weekday`, `is_weekend` |
| `fact_model_summary` | one row per model, whole 180-day window | `mae`, `rmse`, `r2`, `total_profit_eur`, `pct_of_perfect` |
| `fact_model_day` | model x day (2,148) | `profit_eur`, `perfect_profit_eur`, `mae`, `bias`, `pct_of_perfect_day` |
| `fact_hourly_tail` | model x hour, last 14 days | `timestamp`, `predicted`, `actual` |

Relationships (all many-to-one, single direction): each fact's `model_key` to `dim_model`; `fact_model_day[date_key]` to `dim_date`. Mark `dim_date` as the date table.

## Steps

1. **Get data**: Home > Get data > Text/CSV, load the five files from `powerbi/data` (or one Folder query), set `date_key`/`model_key` to whole numbers.
2. **Relationships**: Model view, drag the keys as above. Sort `dim_date[month_name]` by `month` and `weekday` by `weekday_number`.
3. **Measures**: paste `measures.dax` into a new `_Measures` table (the file explains each one).
4. **Pages**
   - *Which model wins?* Clustered bar of `MAE EUR/MWh` by model next to a bar of `Profit EUR` by model, a table with `Rank By MAE`, `Rank By Profit`, `Rank Shift`, and a slicer on `dim_model[family]`. This is the page that shows the project's finding: the most accurate forecasters are not the most profitable.
   - *Value over the naive baseline*: `Profit vs Naive EUR` by model with `Days Beating Naive` as a data label; a card for `Pct Of Perfect`.
   - *Over time*: line chart of `Daily MAE` and `Profit EUR` by `dim_date[date]`, model as legend, a date range slicer, weekend shading via `is_weekend`.
   - *One model up close*: a model slicer (single select), actual vs predicted from `fact_hourly_tail` (`Actual Price`, `Predicted Price` by `timestamp`), `Daily Bias` as a card.
5. **Look**: no theme needed; use one hue for the baselines and one for the neural nets (`family` as legend) so the grouping reads at a glance.

## What the numbers say (from the committed sample)

Random Forest has the lowest error (MAE 28.2 EUR/MWh) and the highest profit (95.1% of perfect foresight); the naive "same hour last week" baseline reaches 94.3% with an MAE of 41.1, and every neural model earns less than the naive baseline despite a better R2. The report is built to make that visible in one glance.

## Limits

The sample is one 180-day window of one market and one battery size; per-day errors for models whose cached forecasts are missing are blank. Daily profit is in EUR for a 1 MW / 4 MWh battery.
