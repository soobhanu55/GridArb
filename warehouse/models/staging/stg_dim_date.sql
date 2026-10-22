select
    cast(date_key as integer) as date_key,
    cast(date as date) as day,
    year,
    month,
    month_name,
    weekday,
    cast(is_weekend as boolean) as is_weekend
from read_csv_auto('{{ var("data_dir") }}/dim_date.csv')
