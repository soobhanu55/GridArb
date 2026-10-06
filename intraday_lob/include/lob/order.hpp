#pragma once

#include "lob/types.hpp"

namespace lob {

struct Order {
    OrderId id{};
    Side side{};
    OrderType type{};
    Price price{};        // ignored for Market orders
    Quantity quantity{};  // remaining, unfilled quantity
    Quantity original_quantity{};
    Timestamp submitted_at{};
    OrderStatus status{OrderStatus::New};

    [[nodiscard]] bool is_fully_filled() const noexcept { return quantity == 0; }
};

struct Fill {
    OrderId resting_order_id{};
    OrderId aggressor_order_id{};
    Price price{};
    Quantity quantity{};
    Side aggressor_side{};
    Timestamp timestamp{};
};

}  // namespace lob
