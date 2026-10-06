#include <gtest/gtest.h>

#include "lob/market_maker.hpp"
#include "lob/order_book.hpp"

using namespace lob;

namespace {

OrderBook book_with_quote(Price bid, Price ask, Quantity qty = 100) {
    OrderBook book;
    book.add_limit_order(Order{.id = 1, .side = Side::Buy, .type = OrderType::Limit, .price = bid, .quantity = qty});
    book.add_limit_order(Order{.id = 2, .side = Side::Sell, .type = OrderType::Limit, .price = ask, .quantity = qty});
    return book;
}

}  // namespace

TEST(MarketMaker, FlatInventoryQuotesSymmetricallyAroundFairValue) {
    MarketMakerConfig config;
    config.base_half_spread_ticks = 10;
    config.inventory_skew_ticks_per_unit = 0.0;  // isolate the spread behavior
    MarketMaker mm(config);

    OrderBook book = book_with_quote(995, 1005);  // microprice == 1000 exactly (equal size both sides)
    const Quote quote = mm.compute_quote(book, /*inventory=*/0, /*volatility=*/0.0);

    const double fair_value = *book.microprice();
    const double bid_dist = fair_value - to_display_price(quote.bid);
    const double ask_dist = to_display_price(quote.ask) - fair_value;
    EXPECT_NEAR(bid_dist, ask_dist, 1e-9);
}

TEST(MarketMaker, LongInventorySkewsQuotesDown) {
    MarketMakerConfig config;
    config.inventory_skew_ticks_per_unit = 0.5;  // exaggerated for a clear, testable effect
    MarketMaker mm(config);

    OrderBook book = book_with_quote(995, 1005);
    const Quote flat_quote = mm.compute_quote(book, /*inventory=*/0, 0.0);
    const Quote long_quote = mm.compute_quote(book, /*inventory=*/100, 0.0);

    // Being long should push both bid and ask DOWN relative to flat,
    // making the market maker less eager to buy more and keener to sell.
    EXPECT_LT(long_quote.bid, flat_quote.bid);
    EXPECT_LT(long_quote.ask, flat_quote.ask);
}

TEST(MarketMaker, HigherVolatilityWidensTheSpread) {
    MarketMakerConfig config;
    config.inventory_skew_ticks_per_unit = 0.0;
    MarketMaker mm(config);

    OrderBook book = book_with_quote(995, 1005);
    const Quote calm_quote = mm.compute_quote(book, 0, /*volatility=*/0.0);
    const Quote volatile_quote = mm.compute_quote(book, 0, /*volatility=*/5.0);

    const Price calm_spread = calm_quote.ask - calm_quote.bid;
    const Price volatile_spread = volatile_quote.ask - volatile_quote.bid;
    EXPECT_GT(volatile_spread, calm_spread);
}

TEST(MarketMaker, StartsFlatWithZeroPnl) {
    MarketMaker mm(MarketMakerConfig{});
    EXPECT_EQ(mm.inventory(), 0);
    EXPECT_DOUBLE_EQ(mm.realized_pnl(), 0.0);
}

TEST(MarketMaker, BuyingFromAFillIncreasesInventoryAndSetsEntryPrice) {
    MarketMaker mm(MarketMakerConfig{});
    // aggressor SELLS -> hits our bid -> we BUY.
    mm.on_fill(Side::Sell, 10, 100.0);

    EXPECT_EQ(mm.inventory(), 10);
    EXPECT_DOUBLE_EQ(mm.avg_entry_price(), 100.0);
    EXPECT_DOUBLE_EQ(mm.realized_pnl(), 0.0);  // nothing closed yet
}

TEST(MarketMaker, AddingToLongPositionBlendsAverageEntryPrice) {
    MarketMaker mm(MarketMakerConfig{});
    mm.on_fill(Side::Sell, 10, 100.0);  // buy 10 @ 100
    mm.on_fill(Side::Sell, 10, 110.0);  // buy 10 more @ 110

    EXPECT_EQ(mm.inventory(), 20);
    EXPECT_DOUBLE_EQ(mm.avg_entry_price(), 105.0);  // (100*10 + 110*10) / 20
}

TEST(MarketMaker, ClosingLongPositionAtAHigherPriceRealizesProfit) {
    MarketMaker mm(MarketMakerConfig{});
    mm.on_fill(Side::Sell, 10, 100.0);  // buy 10 @ 100 (inventory = +10)
    mm.on_fill(Side::Buy, 10, 105.0);   // aggressor buys -> we sell 10 @ 105

    EXPECT_EQ(mm.inventory(), 0);
    EXPECT_DOUBLE_EQ(mm.realized_pnl(), 50.0);  // 10 units * (105 - 100)
}

TEST(MarketMaker, ClosingShortPositionAtALowerPriceRealizesProfit) {
    MarketMaker mm(MarketMakerConfig{});
    mm.on_fill(Side::Buy, 10, 100.0);  // aggressor buys -> we sell 10 (inventory = -10, entry 100)
    mm.on_fill(Side::Sell, 10, 95.0);  // aggressor sells -> we buy 10 @ 95, covering the short

    EXPECT_EQ(mm.inventory(), 0);
    EXPECT_DOUBLE_EQ(mm.realized_pnl(), 50.0);  // short profits when price falls: 10 * (100 - 95)
}

TEST(MarketMaker, FlippingFromLongToShortRealizesOnTheClosedPortionOnly) {
    MarketMaker mm(MarketMakerConfig{});
    mm.on_fill(Side::Sell, 10, 100.0);  // buy 10 @ 100 (inventory +10)
    mm.on_fill(Side::Buy, 15, 110.0);   // sell 15 @ 110: closes 10 long + opens 5 short

    EXPECT_EQ(mm.inventory(), -5);
    EXPECT_DOUBLE_EQ(mm.realized_pnl(), 100.0);       // 10 * (110 - 100) on the closed portion
    EXPECT_DOUBLE_EQ(mm.avg_entry_price(), 110.0);    // new short's entry is this fill's price
}

TEST(MarketMaker, MarkToMarketPnlIncludesUnrealizedComponent) {
    MarketMaker mm(MarketMakerConfig{});
    mm.on_fill(Side::Sell, 10, 100.0);  // buy 10 @ 100, still open

    EXPECT_DOUBLE_EQ(mm.mark_to_market_pnl(110.0), 100.0);  // 10 * (110 - 100), nothing realized yet
    EXPECT_DOUBLE_EQ(mm.mark_to_market_pnl(90.0), -100.0);
}

TEST(MarketMaker, QuotesRespectMaxInventoryCapOnTheBuySide) {
    MarketMakerConfig config;
    config.max_inventory = 50;
    MarketMaker mm(config);

    OrderBook book = book_with_quote(995, 1005);
    const Quote at_cap = mm.compute_quote(book, /*inventory=*/50, 0.0);
    const Quote under_cap = mm.compute_quote(book, /*inventory=*/10, 0.0);

    // At the cap, the bid should be pushed far away (much lower) so it's
    // not realistically going to trade and add to inventory further.
    EXPECT_LT(at_cap.bid, under_cap.bid);
}
