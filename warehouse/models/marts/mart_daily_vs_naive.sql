-- every model day compared with the naive "same hour last week" profit on the same day
with naive as (
    select d.date_key, d.profit_eur as naive_profit_eur
    from {{ ref('stg_model_day') }} d
    join {{ ref('stg_dim_model') }} m using (model_key)
    where m.model = 'naive'
)
select
    d.model_key,
    m.model,
    m.family,
    dt.day,
    dt.year,
    dt.month,
    d.profit_eur,
    d.perfect_profit_eur,
    n.naive_profit_eur,
    d.profit_eur - n.naive_profit_eur as profit_vs_naive_eur,
    d.profit_eur > n.naive_profit_eur as beats_naive
from {{ ref('stg_model_day') }} d
join {{ ref('stg_dim_model') }} m using (model_key)
join {{ ref('stg_dim_date') }} dt using (date_key)
join naive n using (date_key)
