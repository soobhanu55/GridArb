-- no model is missing days: each model has as many rows as the date dimension
select model_key, count(*) as n
from {{ ref('stg_model_day') }}
group by 1
having count(*) <> (select count(*) from {{ ref('stg_dim_date') }})
