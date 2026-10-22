select
    cast(model_key as integer) as model_key,
    model,
    family,
    cast(is_neural as boolean) as is_neural
from read_csv_auto('{{ var("data_dir") }}/dim_model.csv')
