#pragma once

#include "lob/order_book.hpp"
#include "lob/types.hpp"

namespace lob {

struct Quote {
    Price bid{};
    Price ask{};
};

struct MarketMakerConfig {
    // Half-spread quoted around fair value, in ticks.
    Price base_half_spread_ticks = 5;

    // How much each unit of inventory skews the quoted mid away from fair
    // value, in ticks per unit. A positive inventory (long) skews quotes
    // DOWN, making the market maker more eager to sell and less eager to
    // buy more, so inventory mean-reverts toward zero over time.
    double inventory_skew_ticks_per_unit = 0.02;

    // Spread widens with recent realized volatility, since a stale quote
    // in a fast-moving market is adverse-selection risk: informed flow
    // picks it off before the quote can react.
    double volatility_spread_multiplier = 1.5;

    // Hard cap on absolute inventory. Past this, the market maker stops
    // quoting on the side that would increase inventory further.
    Quantity max_inventory = 1000;
};

// A simplified but genuine market maker: it estimates fair value from the
// book's microprice, skews its quoted midpoint away from fair value based
// on current inventory (to encourage mean reversion rather than running an
// unbounded position), and widens its spread when recent volatility is
// high. This is a heuristic model, not a full stochastic-control solution
// (e.g. Avellaneda-Stoikov), and is documented as such rather than
// overclaiming.
class MarketMaker {
public:
    explicit MarketMaker(MarketMakerConfig config) : config_(config) {}

    // Computes the quote this market maker would post given the current
    // book state, its own inventory, and a recent volatility estimate
    // (e.g. a rolling stddev of mid-price returns, in ticks).
    [[nodiscard]] Quote compute_quote(const OrderBook& book, Quantity inventory,
                                       double recent_volatility_ticks) const;

    // Call when an aggressor trades against this market maker's resting
    // quote. `aggressor_side` is the side of the INCOMING order: if it's
    // Buy, the aggressor bought from the market maker's ask (so the
    // market maker sold); if Sell, the aggressor hit the market maker's
    // bid (so the market maker bought). `price` is the display price
    // (not raw ticks) at which the fill occurred.
    void on_fill(Side aggressor_side, Quantity quantity, double price);

    [[nodiscard]] Quantity inventory() const { return inventory_; }
    [[nodiscard]] double realized_pnl() const { return realized_pnl_; }
    [[nodiscard]] double avg_entry_price() const { return avg_entry_price_; }

    // Marks open inventory to a current reference price for a total P&L
    // view (realized + unrealized), since realized P&L alone hides the
    // risk sitting in an open position.
    [[nodiscard]] double mark_to_market_pnl(double reference_price) const;

private:
    MarketMakerConfig config_;
    Quantity inventory_ = 0;
    double avg_entry_price_ = 0.0;
    double realized_pnl_ = 0.0;
};

}  // namespace lob
