{{ config(severity='warn') }}
-- A WARNING, not an error, and the reason is worth knowing: the battery state of charge carries over from day to day (see
-- src/backtest.py), so every strategy has its own state of charge on each morning. A model that starts a day fuller than the
-- perfect-foresight strategy can earn more than it on that single day, so pct_of_perfect_day can exceed 100%. Only totals over
-- the whole window are bounded (assert_total_profit_never_exceeds_perfect_foresight). Expect a handful of days among ~2,000.
select m.model, d.date_key, d.profit_eur, d.perfect_profit_eur, d.profit_eur - d.perfect_profit_eur as excess_eur
from {{ ref('stg_model_day') }} d
join {{ ref('stg_dim_model') }} m using (model_key)
where d.profit_eur > d.perfect_profit_eur + 1e-6
