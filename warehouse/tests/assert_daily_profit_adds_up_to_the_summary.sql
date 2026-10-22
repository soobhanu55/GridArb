-- the per-day facts and the per-model summary come from the same run, so they must agree to within a euro
select s.model_key, s.total_profit_eur, d.daily_total
from {{ ref('stg_model_summary') }} s
join (select model_key, sum(profit_eur) as daily_total from {{ ref('stg_model_day') }} group by 1) d using (model_key)
where abs(s.total_profit_eur - d.daily_total) > 1.0
