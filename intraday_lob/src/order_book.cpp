#include "lob/order_book.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace lob {

namespace {
Timestamp now() { return std::chrono::steady_clock::now(); }
}  // namespace

template <typename BookSide, typename PriceCrosses>
std::vector<Fill> OrderBook::match_against(BookSide& resting_side, Order& incoming,
                                            PriceCrosses price_crosses) {
    std::vector<Fill> fills;

    auto level_it = resting_side.begin();
    while (incoming.quantity > 0 && level_it != resting_side.end() &&
           price_crosses(level_it->first)) {
        auto& queue = level_it->second;

        while (incoming.quantity > 0 && !queue.empty()) {
            Order& resting = queue.front();
            const Quantity traded = std::min(incoming.quantity, resting.quantity);

            fills.push_back(Fill{
                .resting_order_id = resting.id,
                .aggressor_order_id = incoming.id,
                .price = level_it->first,
                .quantity = traded,
                .aggressor_side = incoming.side,
                .timestamp = now(),
            });

            incoming.quantity -= traded;
            resting.quantity -= traded;

            if (resting.is_fully_filled()) {
                resting.status = OrderStatus::Filled;
                order_index_.erase(resting.id);
                queue.pop_front();
            } else {
                resting.status = OrderStatus::PartiallyFilled;
            }
        }

        if (queue.empty()) {
            level_it = resting_side.erase(level_it);
        } else {
            ++level_it;
        }
    }

    return fills;
}

void OrderBook::rest_order(Order order) {
    order.status = OrderStatus::New;
    if (order.side == Side::Buy) {
        auto& queue = bids_[order.price];
        queue.push_back(order);
        order_index_[order.id] = {Side::Buy, order.price, std::prev(queue.end())};
    } else {
        auto& queue = asks_[order.price];
        queue.push_back(order);
        order_index_[order.id] = {Side::Sell, order.price, std::prev(queue.end())};
    }
}

std::vector<Fill> OrderBook::add_limit_order(Order order) {
    order.original_quantity = order.quantity;
    order.type = OrderType::Limit;

    std::vector<Fill> fills;
    if (order.side == Side::Buy) {
        fills = match_against(asks_, order,
                               [&](Price ask_price) { return ask_price <= order.price; });
    } else {
        fills = match_against(bids_, order,
                               [&](Price bid_price) { return bid_price >= order.price; });
    }

    if (order.quantity > 0) {
        rest_order(order);
    }
    return fills;
}

std::vector<Fill> OrderBook::add_market_order(OrderId id, Side side, Quantity quantity) {
    Order order{
        .id = id,
        .side = side,
        .type = OrderType::Market,
        .price = 0,
        .quantity = quantity,
        .original_quantity = quantity,
    };

    if (side == Side::Buy) {
        return match_against(asks_, order, [](Price) { return true; });
    }
    return match_against(bids_, order, [](Price) { return true; });
}

bool OrderBook::cancel_order(OrderId id) {
    auto it = order_index_.find(id);
    if (it == order_index_.end()) {
        return false;
    }

    const auto [side, price, list_it] = it->second;
    if (side == Side::Buy) {
        auto level_it = bids_.find(price);
        if (level_it == bids_.end()) {
            std::cerr << "[FATAL] cancel_order: order_index_ has id=" << id
                      << " at price=" << price << " but bids_ has no such level\n";
            std::abort();
        }
        level_it->second.erase(list_it);
        if (level_it->second.empty()) {
            bids_.erase(level_it);
        }
    } else {
        auto level_it = asks_.find(price);
        if (level_it == asks_.end()) {
            std::cerr << "[FATAL] cancel_order: order_index_ has id=" << id
                      << " at price=" << price << " but asks_ has no such level\n";
            std::abort();
        }
        level_it->second.erase(list_it);
        if (level_it->second.empty()) {
            asks_.erase(level_it);
        }
    }

    order_index_.erase(it);
    return true;
}

bool OrderBook::modify_order(OrderId id, Price new_price, Quantity new_quantity) {
    auto it = order_index_.find(id);
    if (it == order_index_.end()) {
        return false;
    }
    const Side side = it->second.side;

    Order replacement = *it->second.iterator;
    replacement.price = new_price;
    replacement.quantity = new_quantity;
    replacement.original_quantity = new_quantity;

    cancel_order(id);
    replacement.side = side;
    add_limit_order(replacement);
    return true;
}

std::optional<Price> OrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
}

std::optional<Price> OrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

std::optional<Price> OrderBook::spread() const {
    const auto bid = best_bid();
    const auto ask = best_ask();
    if (!bid || !ask) return std::nullopt;
    return *ask - *bid;
}

std::optional<double> OrderBook::mid_price() const {
    const auto bid = best_bid();
    const auto ask = best_ask();
    if (!bid || !ask) return std::nullopt;
    return to_display_price(*bid + *ask) / 2.0;
}

Quantity OrderBook::total_quantity_at_level(const std::list<Order>& level) const {
    Quantity total = 0;
    for (const auto& order : level) total += order.quantity;
    return total;
}

std::optional<double> OrderBook::microprice() const {
    const auto bid = best_bid();
    const auto ask = best_ask();
    if (!bid || !ask) return std::nullopt;

    const Quantity bid_qty = total_quantity_at_level(bids_.begin()->second);
    const Quantity ask_qty = total_quantity_at_level(asks_.begin()->second);
    const Quantity total_qty = bid_qty + ask_qty;
    if (total_qty == 0) return mid_price();

    // Weighted by the OPPOSITE side's quantity: a thick ask relative to
    // the bid means more supply waiting just above, pulling fair value
    // down toward the bid.
    const double weighted =
        (to_display_price(*bid) * static_cast<double>(ask_qty) +
         to_display_price(*ask) * static_cast<double>(bid_qty)) /
        static_cast<double>(total_qty);
    return weighted;
}

Quantity OrderBook::depth_at(Side side, std::size_t level) const {
    if (side == Side::Buy) {
        if (level >= bids_.size()) return 0;
        auto it = bids_.begin();
        std::advance(it, level);
        return total_quantity_at_level(it->second);
    }
    if (level >= asks_.size()) return 0;
    auto it = asks_.begin();
    std::advance(it, level);
    return total_quantity_at_level(it->second);
}

double OrderBook::volume_imbalance(std::size_t levels) const {
    Quantity bid_total = 0;
    Quantity ask_total = 0;
    for (std::size_t i = 0; i < levels; ++i) {
        bid_total += depth_at(Side::Buy, i);
        ask_total += depth_at(Side::Sell, i);
    }
    const Quantity total = bid_total + ask_total;
    if (total == 0) return 0.0;
    return static_cast<double>(bid_total - ask_total) / static_cast<double>(total);
}

std::optional<double> OrderBook::vwap(Side side, Quantity target_quantity) const {
    Quantity remaining = target_quantity;
    double notional = 0.0;

    if (side == Side::Buy) {
        // Buying walks UP the ask side (paying more as you take more liquidity).
        for (const auto& [price, queue] : asks_) {
            if (remaining <= 0) break;
            const Quantity available = total_quantity_at_level(queue);
            const Quantity taken = std::min(remaining, available);
            notional += to_display_price(price) * static_cast<double>(taken);
            remaining -= taken;
        }
    } else {
        // Selling walks DOWN the bid side.
        for (const auto& [price, queue] : bids_) {
            if (remaining <= 0) break;
            const Quantity available = total_quantity_at_level(queue);
            const Quantity taken = std::min(remaining, available);
            notional += to_display_price(price) * static_cast<double>(taken);
            remaining -= taken;
        }
    }

    if (remaining > 0) return std::nullopt;  // not enough resting liquidity
    return notional / static_cast<double>(target_quantity);
}

}  // namespace lob
