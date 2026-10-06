#pragma once

#include <deque>
#include <random>
#include <vector>

#include "lob/market_maker.hpp"
#include "lob/order_book.hpp"
#include "lob/types.hpp"

namespace lob {

struct SimulationConfig {
    int steps = 5000;
    double mid_price_start = 100.0;
    double mid_price_vol_per_step = 0.05;   // stddev of the underlying random-walk price, in currency units
    double arrival_rate_per_step = 0.6;     // probability an aggressor order arrives on a given step
    Quantity aggressor_order_size = 10;
    int volatility_window = 30;             // steps used for the market maker's realized-vol estimate
    std::uint64_t random_seed = 42;
};

struct SimulationStepResult {
    int step{};
    double mid_price{};
    Price mm_bid{};
    Price mm_ask{};
    Quantity mm_inventory{};
    double mm_realized_pnl{};
    double mm_mark_to_market_pnl{};
};

// Drives a synthetic order flow against a book where the market maker is
// always quoting: a reference "true" price random-walks each step, and
// with some probability a market-order aggressor arrives and trades
// against whichever side of the market maker's quote is currently
// favorable relative to that reference price. This is a simplified but
// genuine adverse-selection setup: informed-ish flow tends to hit the
// market maker's stale side after the reference price has moved.
class ExecutionSimulator {
public:
    ExecutionSimulator(SimulationConfig config, MarketMakerConfig mm_config);

    std::vector<SimulationStepResult> run();

private:
    SimulationConfig config_;
    MarketMaker market_maker_;
    OrderBook book_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> price_step_dist_;
    std::uniform_real_distribution<double> arrival_dist_;
    std::deque<double> recent_mid_prices_;
    OrderId next_order_id_ = 1;
    OrderId mm_bid_id_ = 0;
    OrderId mm_ask_id_ = 0;

    double estimate_recent_volatility_ticks() const;
    void refresh_quotes(double true_price);
};

}  // namespace lob
