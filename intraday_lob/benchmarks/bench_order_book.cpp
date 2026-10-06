#include <benchmark/benchmark.h>

#include <random>

#include "lob/order_book.hpp"

using namespace lob;

// Adding a non-crossing limit order (pure insertion into an empty price
// level or an existing one), the most common single operation on a live
// book.
static void BM_AddNonCrossingLimitOrder(benchmark::State& state) {
    OrderBook book;
    OrderId id = 1;
    for (auto _ : state) {
        book.add_limit_order(Order{
            .id = id, .side = Side::Buy, .type = OrderType::Limit,
            .price = 100, .quantity = 10,
        });
        ++id;
        // Cancel periodically so the book doesn't grow unbounded and
        // start measuring map-size effects instead of insertion cost.
        if (id % 100 == 0) {
            for (OrderId c = id - 99; c < id; ++c) book.cancel_order(c);
        }
    }
}
BENCHMARK(BM_AddNonCrossingLimitOrder);

// Cancelling a resting order: O(1) via the order index, should stay flat
// regardless of book depth.
static void BM_CancelOrder(benchmark::State& state) {
    OrderBook book;
    OrderId id = 1;
    for (auto _ : state) {
        state.PauseTiming();
        book.add_limit_order(Order{
            .id = id, .side = Side::Buy, .type = OrderType::Limit,
            .price = 100, .quantity = 10,
        });
        state.ResumeTiming();
        book.cancel_order(id);
        ++id;
    }
}
BENCHMARK(BM_CancelOrder);

// best_bid()/best_ask() lookups: should be O(1), just reading the first
// element of an already-sorted map.
static void BM_BestBidAsk(benchmark::State& state) {
    OrderBook book;
    for (int i = 0; i < 1000; ++i) {
        book.add_limit_order(Order{
            .id = static_cast<OrderId>(i + 1), .side = Side::Buy, .type = OrderType::Limit,
            .price = 100 - i, .quantity = 10,
        });
        book.add_limit_order(Order{
            .id = static_cast<OrderId>(i + 2000), .side = Side::Sell, .type = OrderType::Limit,
            .price = 200 + i, .quantity = 10,
        });
    }
    for (auto _ : state) {
        benchmark::DoNotOptimize(book.best_bid());
        benchmark::DoNotOptimize(book.best_ask());
    }
}
BENCHMARK(BM_BestBidAsk);

// A crossing market order that walks and consumes several price levels,
// the most expensive single operation the book performs.
static void BM_MarketOrderWalkingFiveLevels(benchmark::State& state) {
    OrderId id = 1;
    for (auto _ : state) {
        state.PauseTiming();
        OrderBook book;
        for (int level = 0; level < 5; ++level) {
            book.add_limit_order(Order{
                .id = static_cast<OrderId>(id++), .side = Side::Sell, .type = OrderType::Limit,
                .price = 100 + level, .quantity = 10,
            });
        }
        state.ResumeTiming();
        benchmark::DoNotOptimize(book.add_market_order(id++, Side::Buy, 50));
    }
}
BENCHMARK(BM_MarketOrderWalkingFiveLevels);

// Throughput under a realistic mixed workload: mostly non-crossing
// inserts with occasional cancels and occasional crossing orders, closer
// to what a real feed handler sees than any single isolated operation.
static void BM_MixedWorkload(benchmark::State& state) {
    OrderBook book;
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> op_dist(0, 9);
    std::uniform_int_distribution<Price> price_dist(90, 110);
    OrderId id = 1;
    std::vector<OrderId> resting_ids;

    for (auto _ : state) {
        const int op = op_dist(rng);
        if (op < 6 || resting_ids.empty()) {
            // 60%: add a limit order
            const Side side = (op % 2 == 0) ? Side::Buy : Side::Sell;
            const Price price = (side == Side::Buy) ? price_dist(rng) - 20 : price_dist(rng) + 20;
            book.add_limit_order(Order{.id = id, .side = side, .type = OrderType::Limit,
                                        .price = price, .quantity = 10});
            resting_ids.push_back(id);
            ++id;
        } else if (op < 8) {
            // 20%: cancel a random resting order
            const auto idx = rng() % resting_ids.size();
            book.cancel_order(resting_ids[idx]);
            resting_ids.erase(resting_ids.begin() + static_cast<long>(idx));
        } else {
            // 20%: a market order
            const Side side = (op % 2 == 0) ? Side::Buy : Side::Sell;
            benchmark::DoNotOptimize(book.add_market_order(id++, side, 5));
        }
    }
}
BENCHMARK(BM_MixedWorkload);

BENCHMARK_MAIN();
