#include <gtest/gtest.h>

#include "lob/order_book.hpp"

using namespace lob;

namespace {

Order make_limit(OrderId id, Side side, Price price, Quantity qty) {
    return Order{.id = id, .side = side, .type = OrderType::Limit, .price = price, .quantity = qty};
}

}  // namespace

TEST(OrderBook, EmptyBookHasNoBidAskOrSpread) {
    OrderBook book;
    EXPECT_FALSE(book.best_bid().has_value());
    EXPECT_FALSE(book.best_ask().has_value());
    EXPECT_FALSE(book.spread().has_value());
    EXPECT_FALSE(book.mid_price().has_value());
}

TEST(OrderBook, RestingLimitOrderBecomesBestBidOrAsk) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Buy, 100, 10));
    book.add_limit_order(make_limit(2, Side::Sell, 105, 10));

    EXPECT_EQ(book.best_bid(), 100);
    EXPECT_EQ(book.best_ask(), 105);
    EXPECT_EQ(book.spread(), 5);
    EXPECT_DOUBLE_EQ(*book.mid_price(), (to_display_price(100) + to_display_price(105)) / 2.0);
}

TEST(OrderBook, BidOrderingIsDescendingAskOrderingIsAscending) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Buy, 100, 5));
    book.add_limit_order(make_limit(2, Side::Buy, 102, 5));   // better bid
    book.add_limit_order(make_limit(3, Side::Buy, 98, 5));
    book.add_limit_order(make_limit(4, Side::Sell, 110, 5));
    book.add_limit_order(make_limit(5, Side::Sell, 107, 5));  // better ask
    book.add_limit_order(make_limit(6, Side::Sell, 112, 5));

    EXPECT_EQ(book.best_bid(), 102);
    EXPECT_EQ(book.best_ask(), 107);
}

TEST(OrderBook, CrossingLimitOrderMatchesImmediately) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Sell, 100, 10));

    const auto fills = book.add_limit_order(make_limit(2, Side::Buy, 100, 10));

    ASSERT_EQ(fills.size(), 1u);
    EXPECT_EQ(fills[0].resting_order_id, 1u);
    EXPECT_EQ(fills[0].aggressor_order_id, 2u);
    EXPECT_EQ(fills[0].quantity, 10);
    EXPECT_EQ(fills[0].price, 100);
    // fully matched, nothing rests
    EXPECT_FALSE(book.best_bid().has_value());
    EXPECT_FALSE(book.best_ask().has_value());
}

TEST(OrderBook, PartialFillLeavesRemainderResting) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Sell, 100, 10));

    const auto fills = book.add_limit_order(make_limit(2, Side::Buy, 100, 15));

    ASSERT_EQ(fills.size(), 1u);
    EXPECT_EQ(fills[0].quantity, 10);
    // 5 units of the buy order should now rest as the best bid
    EXPECT_EQ(book.best_bid(), 100);
    EXPECT_FALSE(book.best_ask().has_value());
}

TEST(OrderBook, FifoWithinAPriceLevel) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Buy, 100, 5));
    book.add_limit_order(make_limit(2, Side::Buy, 100, 5));  // arrives second at the same price

    // A market sell for 5 should match order 1 (first in) entirely, not order 2.
    const auto fills = book.add_market_order(3, Side::Sell, 5);

    ASSERT_EQ(fills.size(), 1u);
    EXPECT_EQ(fills[0].resting_order_id, 1u);
    // order 2 should still be resting for the full 5
    EXPECT_EQ(book.depth_at(Side::Buy, 0), 5);
}

TEST(OrderBook, MarketOrderWalksMultiplePriceLevels) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Sell, 100, 5));
    book.add_limit_order(make_limit(2, Side::Sell, 101, 5));
    book.add_limit_order(make_limit(3, Side::Sell, 102, 5));

    const auto fills = book.add_market_order(4, Side::Buy, 12);

    ASSERT_EQ(fills.size(), 3u);
    EXPECT_EQ(fills[0].price, 100);
    EXPECT_EQ(fills[0].quantity, 5);
    EXPECT_EQ(fills[1].price, 101);
    EXPECT_EQ(fills[1].quantity, 5);
    EXPECT_EQ(fills[2].price, 102);
    EXPECT_EQ(fills[2].quantity, 2);
    // 3 units remain resting at 102
    EXPECT_EQ(book.best_ask(), 102);
    EXPECT_EQ(book.depth_at(Side::Sell, 0), 3);
}

TEST(OrderBook, CancelRemovesRestingOrder) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Buy, 100, 10));

    EXPECT_TRUE(book.cancel_order(1));
    EXPECT_FALSE(book.best_bid().has_value());
    EXPECT_FALSE(book.cancel_order(1));  // already gone
}

TEST(OrderBook, CancelOnlyRemovesTheTargetOrderFromASharedLevel) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Buy, 100, 5));
    book.add_limit_order(make_limit(2, Side::Buy, 100, 7));

    EXPECT_TRUE(book.cancel_order(1));
    EXPECT_EQ(book.depth_at(Side::Buy, 0), 7);  // only order 2's quantity remains
}

TEST(OrderBook, ModifyPriceMovesOrderToNewLevelAndLosesPriority) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Buy, 100, 5));
    book.add_limit_order(make_limit(2, Side::Buy, 100, 5));

    EXPECT_TRUE(book.modify_order(1, 101, 5));

    EXPECT_EQ(book.best_bid(), 101);
    EXPECT_EQ(book.depth_at(Side::Buy, 0), 5);   // order 1 alone at 101
    EXPECT_EQ(book.depth_at(Side::Buy, 1), 5);   // order 2 alone left at 100
}

TEST(OrderBook, DepthAtAggregatesMultipleOrdersOnSameLevel) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Buy, 100, 5));
    book.add_limit_order(make_limit(2, Side::Buy, 100, 7));

    EXPECT_EQ(book.depth_at(Side::Buy, 0), 12);
}

TEST(OrderBook, DepthAtBeyondBookSizeReturnsZero) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Buy, 100, 5));
    EXPECT_EQ(book.depth_at(Side::Buy, 3), 0);
}

TEST(OrderBook, VolumeImbalanceIsPositiveWhenBidHeavy) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Buy, 100, 30));
    book.add_limit_order(make_limit(2, Side::Sell, 101, 10));

    const double imbalance = book.volume_imbalance(1);
    EXPECT_DOUBLE_EQ(imbalance, (30.0 - 10.0) / 40.0);
}

TEST(OrderBook, VolumeImbalanceIsZeroOnEmptyBook) {
    OrderBook book;
    EXPECT_DOUBLE_EQ(book.volume_imbalance(1), 0.0);
}

TEST(OrderBook, MicropriceWeightsTowardThinnerOppositeSide) {
    OrderBook book;
    // Thick ask, thin bid: microprice should sit closer to the bid.
    book.add_limit_order(make_limit(1, Side::Buy, 100, 5));
    book.add_limit_order(make_limit(2, Side::Sell, 102, 95));

    const auto micro = book.microprice();
    ASSERT_TRUE(micro.has_value());
    const double mid = to_display_price(101);
    EXPECT_LT(*micro, mid);
}

TEST(OrderBook, VwapWalksBookAndAveragesCorrectly) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Sell, 100, 5));
    book.add_limit_order(make_limit(2, Side::Sell, 102, 5));

    // Buying 10 total: 5 at 100 + 5 at 102 = notional 1010, / 10 = 101.0
    const auto vwap = book.vwap(Side::Buy, 10);
    ASSERT_TRUE(vwap.has_value());
    EXPECT_DOUBLE_EQ(*vwap, to_display_price(101));
}

TEST(OrderBook, VwapReturnsNulloptWhenInsufficientLiquidity) {
    OrderBook book;
    book.add_limit_order(make_limit(1, Side::Sell, 100, 5));

    EXPECT_FALSE(book.vwap(Side::Buy, 100).has_value());
}

TEST(OrderBook, OrderCountTracksRestingOrders) {
    OrderBook book;
    EXPECT_EQ(book.order_count(), 0u);
    book.add_limit_order(make_limit(1, Side::Buy, 100, 5));
    book.add_limit_order(make_limit(2, Side::Sell, 105, 5));
    EXPECT_EQ(book.order_count(), 2u);
    book.cancel_order(1);
    EXPECT_EQ(book.order_count(), 1u);
}
