#include "lob/market_maker.hpp"

#include <algorithm>
#include <cmath>

namespace lob {

Quote MarketMaker::compute_quote(const OrderBook& book, Quantity inventory,
                                  double recent_volatility_ticks) const {
    const auto fair_value = book.microprice();
    const double fv = fair_value.value_or(0.0);

    // Skew the quoted midpoint away from fair value based on inventory:
    // a positive (long) inventory pushes the midpoint down, making the
    // ask more attractive to hit and the bid less attractive, so the
    // market maker is more likely to sell down toward flat.
    const double skew =
        static_cast<double>(inventory) * config_.inventory_skew_ticks_per_unit * kTickSize;
    const double skewed_mid = fv - skew;

    // Widen around a volatile market: a resting quote is more likely to
    // get picked off by informed flow when the market is moving fast.
    const double half_spread =
        static_cast<double>(config_.base_half_spread_ticks) * kTickSize *
        (1.0 + config_.volatility_spread_multiplier * recent_volatility_ticks);

    Price bid = from_display_price(skewed_mid - half_spread);
    Price ask = from_display_price(skewed_mid + half_spread);

    // Stop adding to a position past the configured cap: pull the quote
    // on the side that would increase inventory further by pricing it
    // far away (never realistically hit), rather than crossing the book.
    if (inventory >= config_.max_inventory) {
        bid = from_display_price(skewed_mid - half_spread * 100.0);
    }
    if (inventory <= -config_.max_inventory) {
        ask = from_display_price(skewed_mid + half_spread * 100.0);
    }

    return Quote{bid, ask};
}

void MarketMaker::on_fill(Side aggressor_side, Quantity quantity, double price) {
    // aggressor buys -> hits our ask -> we sell. aggressor sells -> hits
    // our bid -> we buy.
    const bool we_are_buying = (aggressor_side == Side::Sell);
    const Quantity signed_qty = we_are_buying ? quantity : -quantity;

    if (inventory_ == 0) {
        inventory_ = signed_qty;
        avg_entry_price_ = price;
        return;
    }

    const bool same_direction = (inventory_ > 0) == (signed_qty > 0);

    if (same_direction) {
        // Adding to an existing position: blend the average entry price.
        const double total_qty = static_cast<double>(std::abs(inventory_) + std::abs(signed_qty));
        avg_entry_price_ = (avg_entry_price_ * static_cast<double>(std::abs(inventory_)) +
                             price * static_cast<double>(std::abs(signed_qty))) /
                            total_qty;
        inventory_ += signed_qty;
        return;
    }

    // Reducing or flipping an existing position. Capture the ORIGINAL
    // direction before mutating inventory_, since that's what determines
    // both the P&L sign and whether this fill flips past zero.
    const bool was_long = inventory_ > 0;
    const Quantity closing_qty = std::min(std::abs(inventory_), std::abs(signed_qty));
    const double pnl_per_unit = was_long ? (price - avg_entry_price_) : (avg_entry_price_ - price);
    realized_pnl_ += pnl_per_unit * static_cast<double>(closing_qty);

    inventory_ += signed_qty;

    if (inventory_ == 0) {
        avg_entry_price_ = 0.0;
    } else if ((inventory_ > 0) != was_long) {
        // Sign flipped relative to the ORIGINAL position: the new
        // position's entry price is this fill's price, not a blend with
        // the now-fully-closed old position.
        avg_entry_price_ = price;
    }
    // else: partially reduced but same original direction retained ->
    // avg_entry_price_ is unchanged, which is correct (the remaining
    // shares' cost basis doesn't change when you sell some of them).
}

double MarketMaker::mark_to_market_pnl(double reference_price) const {
    const double unrealized =
        static_cast<double>(inventory_) * (reference_price - avg_entry_price_);
    return realized_pnl_ + unrealized;
}

}  // namespace lob
