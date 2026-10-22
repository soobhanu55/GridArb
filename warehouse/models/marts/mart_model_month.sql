-- monthly profit and share of perfect foresight per model (for the over-time view)
select
    model_key, model, family, year, month,
    count(*) as days,
    sum(profit_eur) as profit_eur,
    sum(profit_eur) / sum(perfect_profit_eur) as pct_of_perfect,
    avg(beats_naive::int) as share_of_days_beating_naive
from {{ ref('mart_daily_vs_naive') }}
group by 1, 2, 3, 4, 5
