#pragma once

#include <list>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#include "lob/order.hpp"
#include "lob/types.hpp"

namespace lob {

// A price-time-priority limit order book for a single instrument.
//
// Bids are kept in descending price order (best bid first), asks in
// ascending price order (best ask first). Within a price level, orders are
// FIFO (first in, first matched), which is how real exchanges allocate
// fills at a given price. Cancel/lookup is O(1) via an index of order id to
// its exact location, rather than scanning price levels.
class OrderBook {
public:
    // Adds a limit order. It matches immediately against any crossing
    // orders on the opposite side (price-time priority), and any
    // unfilled remainder rests in the book. Returns the fills generated.
    std::vector<Fill> add_limit_order(Order order);

    // A market order matches immediately against the best available
    // prices on the opposite side until fully filled or the book on that
    // side is exhausted. It never rests in the book.
    std::vector<Fill> add_market_order(OrderId id, Side side, Quantity quantity);

    // Removes a resting order entirely. Returns false if the id doesn't
    // exist (already filled, already cancelled, or never existed).
    bool cancel_order(OrderId id);

    // Exchange semantics: a price change loses time priority (the order
    // is cancelled and re-inserted at the back of the new price level's
    // queue), matching how real order-driven markets treat modifies.
    bool modify_order(OrderId id, Price new_price, Quantity new_quantity);

    [[nodiscard]] std::optional<Price> best_bid() const;
    [[nodiscard]] std::optional<Price> best_ask() const;
    [[nodiscard]] std::optional<Price> spread() const;
    [[nodiscard]] std::optional<double> mid_price() const;

    // Volume-weighted midpoint between best bid and ask, weighted by the
    // *opposite* side's size: a large ask sitting on top of a thin bid
    // pulls the microprice down toward the bid, since that ask is more
    // likely to be the side that gets run through next. This is a better
    // short-horizon fair-value estimate than the plain mid-price.
    [[nodiscard]] std::optional<double> microprice() const;

    // Total resting quantity at the N-th best price level on a side
    // (level 0 = best price). Returns 0 if that level doesn't exist.
    [[nodiscard]] Quantity depth_at(Side side, std::size_t level) const;

    // (bid_volume - ask_volume) / (bid_volume + ask_volume) summed over
    // the top `levels` price levels on each side. Positive means more
    // resting buy pressure than sell pressure near the touch.
    [[nodiscard]] double volume_imbalance(std::size_t levels = 1) const;

    // Volume-weighted average price to fill `target_quantity` by walking
    // the book on `side`, without actually matching anything. Returns
    // nullopt if the side doesn't have enough resting quantity.
    [[nodiscard]] std::optional<double> vwap(Side side, Quantity target_quantity) const;

    [[nodiscard]] std::size_t order_count() const { return order_index_.size(); }

private:
    struct OrderLocation {
        Side side;
        Price price;
        std::list<Order>::iterator iterator;
    };

    // Descending for bids (highest price = best), ascending for asks
    // (lowest price = best) — both achieved by the map's natural
    // ordering plus the comparator on the bid side.
    std::map<Price, std::list<Order>, std::greater<>> bids_;
    std::map<Price, std::list<Order>> asks_;
    std::unordered_map<OrderId, OrderLocation> order_index_;

    template <typename BookSide, typename PriceCrosses>
    std::vector<Fill> match_against(BookSide& resting_side, Order& incoming,
                                     PriceCrosses price_crosses);

    void rest_order(Order order);
    Quantity total_quantity_at_level(const std::list<Order>& level) const;
};

}  // namespace lob
