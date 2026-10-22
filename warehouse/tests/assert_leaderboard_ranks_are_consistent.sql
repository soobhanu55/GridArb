-- no model may be ranked above another one that earned more
select a.model, b.model as outranked_model
from {{ ref('mart_model_leaderboard') }} a
join {{ ref('mart_model_leaderboard') }} b on a.rank_by_profit < b.rank_by_profit
where a.total_profit_eur < b.total_profit_eur
