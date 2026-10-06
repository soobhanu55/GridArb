#include "lob/execution_simulator.hpp"

#include <cmath>
#include <numeric>

namespace lob {

ExecutionSimulator::ExecutionSimulator(SimulationConfig config, MarketMakerConfig mm_config)
    : config_(config),
      market_maker_(mm_config),
      rng_(config.random_seed),
      price_step_dist_(0.0, config.mid_price_vol_per_step),
      arrival_dist_(0.0, 1.0) {}

double ExecutionSimulator::estimate_recent_volatility_ticks() const {
    if (recent_mid_prices_.size() < 2) return 0.0;

    std::vector<double> returns;
    returns.reserve(recent_mid_prices_.size() - 1);
    for (std::size_t i = 1; i < recent_mid_prices_.size(); ++i) {
        returns.push_back(recent_mid_prices_[i] - recent_mid_prices_[i - 1]);
    }

    const double mean = std::accumulate(returns.begin(), returns.end(), 0.0) /
                         static_cast<double>(returns.size());
    double variance = 0.0;
    for (double r : returns) variance += (r - mean) * (r - mean);
    variance /= static_cast<double>(returns.size());

    return std::sqrt(variance) / kTickSize;
}

void ExecutionSimulator::refresh_quotes(double true_price) {
    if (mm_bid_id_ != 0) book_.cancel_order(mm_bid_id_);
    if (mm_ask_id_ != 0) book_.cancel_order(mm_ask_id_);

    // Fair-value input to the quote engine: once the book has resting
    // orders the microprice reflects them, but before the first quote is
    // posted there's nothing resting yet, so seed it with the true price.
    const auto microprice = book_.microprice();
    const double volatility_ticks = estimate_recent_volatility_ticks();

    Quote quote;
    if (microprice.has_value()) {
        quote = market_maker_.compute_quote(book_, market_maker_.inventory(), volatility_ticks);
    } else {
        // No resting liquidity yet to derive a microprice from: quote
        // directly around the true price for this first step only.
        const Price mid_ticks = from_display_price(true_price);
        const Price half_spread = 5;
        quote = Quote{mid_ticks - half_spread, mid_ticks + half_spread};
    }

    mm_bid_id_ = next_order_id_++;
    mm_ask_id_ = next_order_id_++;
    book_.add_limit_order(Order{
        .id = mm_bid_id_, .side = Side::Buy, .type = OrderType::Limit,
        .price = quote.bid, .quantity = config_.aggressor_order_size * 5,
    });
    book_.add_limit_order(Order{
        .id = mm_ask_id_, .side = Side::Sell, .type = OrderType::Limit,
        .price = quote.ask, .quantity = config_.aggressor_order_size * 5,
    });
}

std::vector<SimulationStepResult> ExecutionSimulator::run() {
    std::vector<SimulationStepResult> results;
    results.reserve(static_cast<std::size_t>(config_.steps));

    double true_price = config_.mid_price_start;

    for (int step = 0; step < config_.steps; ++step) {
        true_price += price_step_dist_(rng_);
        true_price = std::max(true_price, 0.01);

        refresh_quotes(true_price);

        const auto bid = book_.best_bid();
        const auto ask = book_.best_ask();

        if (arrival_dist_(rng_) < config_.arrival_rate_per_step && bid && ask) {
            // Bias the aggressor's direction toward the side that's
            // favorable relative to the true price: if the true price has
            // drifted above the market maker's ask, a buyer aggressing
            // the ask is "informed" and adversely selects the MM. If it's
            // genuinely between bid/ask, direction is a coin flip.
            const double ask_display = to_display_price(*ask);
            const double bid_display = to_display_price(*bid);

            Side aggressor_side;
            if (true_price > ask_display) {
                aggressor_side = Side::Buy;
            } else if (true_price < bid_display) {
                aggressor_side = Side::Sell;
            } else {
                aggressor_side = (arrival_dist_(rng_) < 0.5) ? Side::Buy : Side::Sell;
            }

            const OrderId aggressor_id = next_order_id_++;
            const auto fills = book_.add_market_order(aggressor_id, aggressor_side,
                                                        config_.aggressor_order_size);
            for (const auto& fill : fills) {
                market_maker_.on_fill(aggressor_side, fill.quantity, to_display_price(fill.price));
            }
        }

        recent_mid_prices_.push_back(true_price);
        if (static_cast<int>(recent_mid_prices_.size()) > config_.volatility_window) {
            recent_mid_prices_.pop_front();
        }

        const auto refreshed_bid = book_.best_bid();
        const auto refreshed_ask = book_.best_ask();
        results.push_back(SimulationStepResult{
            .step = step,
            .mid_price = true_price,
            .mm_bid = refreshed_bid.value_or(0),
            .mm_ask = refreshed_ask.value_or(0),
            .mm_inventory = market_maker_.inventory(),
            .mm_realized_pnl = market_maker_.realized_pnl(),
            .mm_mark_to_market_pnl = market_maker_.mark_to_market_pnl(true_price),
        });
    }

    return results;
}

}  // namespace lob
