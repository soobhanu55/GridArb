-- one row per model: forecast error, trading result, and how the two rankings disagree (the point of the project)
with base as (
    select
        m.model_key, m.model, m.family, m.is_neural,
        s.mae, s.rmse, s.r2, s.total_profit_eur, s.perfect_profit_eur, s.pct_of_perfect
    from {{ ref('stg_model_summary') }} s
    join {{ ref('stg_dim_model') }} m using (model_key)
),
naive as (select total_profit_eur as naive_profit_eur from base where model = 'naive'),
days as (
    select model_key, count(*) as test_days, sum(case when beats_naive then 1 else 0 end) as days_beating_naive
    from {{ ref('mart_daily_vs_naive') }}
    group by 1
)
select
    b.*,
    rank() over (order by b.mae) as rank_by_mae,
    rank() over (order by b.total_profit_eur desc) as rank_by_profit,
    rank() over (order by b.mae) - rank() over (order by b.total_profit_eur desc) as rank_shift,
    b.total_profit_eur - n.naive_profit_eur as profit_vs_naive_eur,
    d.test_days,
    d.days_beating_naive,
    d.days_beating_naive * 1.0 / d.test_days as share_of_days_beating_naive
from base b
cross join naive n
join days d using (model_key)
order by rank_by_profit
