#pragma once

#include <cstdint>
#include <chrono>

namespace lob {

// Prices are represented as integer ticks, not floating point. A double
// mid-price accumulates rounding error across millions of updates and can
// produce a bid that's numerically greater than the ask it should be below.
// Ticks avoid that class of bug entirely; convert to a display price only
// at the edge (see to_display_price below).
using Price = std::int64_t;
using Quantity = std::int64_t;
using OrderId = std::uint64_t;
using Timestamp = std::chrono::steady_clock::time_point;

// One tick = 0.01 currency units by default. Kept as a named constant
// instead of a magic number so callers can see exactly what a Price means.
inline constexpr double kTickSize = 0.01;

inline double to_display_price(Price ticks) {
    return static_cast<double>(ticks) * kTickSize;
}

inline Price from_display_price(double price) {
    return static_cast<Price>(price / kTickSize + (price >= 0 ? 0.5 : -0.5));
}

enum class Side : std::uint8_t { Buy, Sell };

inline Side opposite(Side side) {
    return side == Side::Buy ? Side::Sell : Side::Buy;
}

enum class OrderType : std::uint8_t { Limit, Market };

enum class OrderStatus : std::uint8_t {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
};

}  // namespace lob
