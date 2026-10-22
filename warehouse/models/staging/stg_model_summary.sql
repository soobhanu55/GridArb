-- one row per model for the whole 180-day test window; pct_of_perfect is stored as a fraction here (the export has percent)
select
    cast(model_key as integer) as model_key,
    mae,
    rmse,
    r2,
    total_profit_eur,
    perfect_profit_eur,
    pct_of_perfect / 100.0 as pct_of_perfect
from read_csv_auto('{{ var("data_dir") }}/fact_model_summary.csv')
