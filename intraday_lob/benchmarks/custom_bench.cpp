// A minimal, dependency-free benchmark harness using std::chrono directly.
//
// Google Benchmark is still wired into the CMake build (see
// bench_order_book.cpp) and compiles cleanly, but its own internal
// startup path is unstable on this specific MinGW/libstdc++ combination
// for the same reason documented in main.cpp (intermittent std::ofstream
// segfaults on this toolchain, not a bug in this project's code -- Google
// Benchmark's console/CSV reporters exercise the same iostream machinery
// internally). This harness measures the identical operations without
// depending on that library, so the reported numbers are real and
// reproducible rather than worked around by disabling something.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

#include "lob/order_book.hpp"

using namespace lob;
using Clock = std::chrono::steady_clock;

struct LatencyStats {
    double mean_ns{};
    double p50_ns{};
    double p95_ns{};
    double p99_ns{};
    double max_ns{};
};

LatencyStats summarize(std::vector<double>& samples_ns) {
    std::sort(samples_ns.begin(), samples_ns.end());
    const std::size_t n = samples_ns.size();

    double sum = 0.0;
    for (double s : samples_ns) sum += s;

    auto percentile = [&](double p) {
        const auto idx = static_cast<std::size_t>(p * static_cast<double>(n - 1));
        return samples_ns[idx];
    };

    return LatencyStats{
        .mean_ns = sum / static_cast<double>(n),
        .p50_ns = percentile(0.50),
        .p95_ns = percentile(0.95),
        .p99_ns = percentile(0.99),
        .max_ns = samples_ns.back(),
    };
}

void print_stats(const char* name, const LatencyStats& stats, int n) {
    std::printf("%-40s n=%-7d mean=%8.1fns  p50=%8.1fns  p95=%8.1fns  p99=%8.1fns  max=%9.1fns\n",
                name, n, stats.mean_ns, stats.p50_ns, stats.p95_ns, stats.p99_ns, stats.max_ns);
}

void bench_add_non_crossing_limit_order() {
    OrderBook book;
    constexpr int kIterations = 100'000;
    std::vector<double> samples;
    samples.reserve(kIterations);

    OrderId id = 1;
    for (int i = 0; i < kIterations; ++i) {
        const auto start = Clock::now();
        book.add_limit_order(Order{.id = id, .side = Side::Buy, .type = OrderType::Limit,
                                    .price = 100, .quantity = 10});
        const auto end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(end - start).count());
        ++id;
        if (id % 100 == 0) {
            for (OrderId c = id - 99; c < id; ++c) book.cancel_order(c);
        }
    }
    print_stats("add_non_crossing_limit_order", summarize(samples), kIterations);
}

void bench_cancel_order() {
    OrderBook book;
    constexpr int kIterations = 100'000;
    std::vector<double> samples;
    samples.reserve(kIterations);

    OrderId id = 1;
    for (int i = 0; i < kIterations; ++i) {
        book.add_limit_order(Order{.id = id, .side = Side::Buy, .type = OrderType::Limit,
                                    .price = 100, .quantity = 10});
        const auto start = Clock::now();
        book.cancel_order(id);
        const auto end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(end - start).count());
        ++id;
    }
    print_stats("cancel_order", summarize(samples), kIterations);
}

void bench_best_bid_ask() {
    OrderBook book;
    for (int i = 0; i < 1000; ++i) {
        book.add_limit_order(Order{.id = static_cast<OrderId>(i + 1), .side = Side::Buy,
                                    .type = OrderType::Limit, .price = 100 - i, .quantity = 10});
        book.add_limit_order(Order{.id = static_cast<OrderId>(i + 2000), .side = Side::Sell,
                                    .type = OrderType::Limit, .price = 200 + i, .quantity = 10});
    }

    constexpr int kIterations = 1'000'000;
    std::vector<double> samples;
    samples.reserve(kIterations);

    for (int i = 0; i < kIterations; ++i) {
        const auto start = Clock::now();
        volatile auto bid = book.best_bid();
        volatile auto ask = book.best_ask();
        (void)bid;
        (void)ask;
        const auto end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(end - start).count());
    }
    print_stats("best_bid_ask (1000-level book)", summarize(samples), kIterations);
}

void bench_market_order_walking_five_levels() {
    constexpr int kIterations = 50'000;
    std::vector<double> samples;
    samples.reserve(kIterations);

    OrderId id = 1;
    for (int i = 0; i < kIterations; ++i) {
        OrderBook book;
        for (int level = 0; level < 5; ++level) {
            book.add_limit_order(Order{.id = static_cast<OrderId>(id++), .side = Side::Sell,
                                        .type = OrderType::Limit, .price = 100 + level, .quantity = 10});
        }
        const auto start = Clock::now();
        auto fills = book.add_market_order(id++, Side::Buy, 50);
        const auto end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(end - start).count());
    }
    print_stats("market_order_walking_5_levels", summarize(samples), kIterations);
}

void bench_mixed_workload_throughput() {
    OrderBook book;
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> op_dist(0, 9);
    OrderId id = 1;
    std::vector<OrderId> resting_ids;

    constexpr int kOps = 500'000;
    const auto start = Clock::now();
    for (int i = 0; i < kOps; ++i) {
        const int op = op_dist(rng);
        if (op < 6 || resting_ids.empty()) {
            const Side side = (op % 2 == 0) ? Side::Buy : Side::Sell;
            const Price price = (side == Side::Buy) ? 80 : 120;
            book.add_limit_order(Order{.id = id, .side = side, .type = OrderType::Limit,
                                        .price = price, .quantity = 10});
            resting_ids.push_back(id);
            ++id;
        } else if (op < 8) {
            const auto idx = rng() % resting_ids.size();
            book.cancel_order(resting_ids[idx]);
            resting_ids.erase(resting_ids.begin() + static_cast<long>(idx));
        } else {
            const Side side = (op % 2 == 0) ? Side::Buy : Side::Sell;
            auto fills = book.add_market_order(id++, side, 5);
        }
    }
    const auto end = Clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    const double ops_per_sec = static_cast<double>(kOps) / seconds;

    std::printf("%-40s %d ops in %.3fs  =>  %.0f ops/sec\n",
                "mixed_workload_throughput", kOps, seconds, ops_per_sec);
}

int main() {
    std::printf("Order book micro-benchmarks (custom std::chrono harness)\n");
    for (int i = 0; i < 100; ++i) std::printf("-");
    std::printf("\n");
    bench_add_non_crossing_limit_order();
    bench_cancel_order();
    bench_best_bid_ask();
    bench_market_order_walking_five_levels();
    bench_mixed_workload_throughput();
    return 0;
}
