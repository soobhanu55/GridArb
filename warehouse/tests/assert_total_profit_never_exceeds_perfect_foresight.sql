-- over the whole test window no forecast-driven strategy can earn more than the perfect-foresight one
select m.model, s.total_profit_eur, s.perfect_profit_eur
from {{ ref('stg_model_summary') }} s
join {{ ref('stg_dim_model') }} m using (model_key)
where s.total_profit_eur > s.perfect_profit_eur + 1e-6
