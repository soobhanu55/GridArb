-- one row per model and test day; mae and bias are null for models whose cached forecasts were missing
select
    cast(model_key as integer) as model_key,
    cast(date_key as integer) as date_key,
    profit_eur,
    perfect_profit_eur,
    mae,
    bias
from read_csv_auto('{{ var("data_dir") }}/fact_model_day.csv')
