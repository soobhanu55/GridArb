#include <cstdio>
#include <iostream>

#include "lob/execution_simulator.hpp"

int main() {
    lob::SimulationConfig sim_config;
    sim_config.steps = 5000;
    sim_config.mid_price_start = 100.0;
    sim_config.mid_price_vol_per_step = 0.05;
    sim_config.arrival_rate_per_step = 0.6;
    sim_config.aggressor_order_size = 10;

    lob::MarketMakerConfig mm_config;
    mm_config.base_half_spread_ticks = 5;
    mm_config.inventory_skew_ticks_per_unit = 0.02;
    mm_config.volatility_spread_multiplier = 1.5;
    mm_config.max_inventory = 1000;

    lob::ExecutionSimulator sim(sim_config, mm_config);
    const auto results = sim.run();

    // Uses the C stdio API rather than std::ofstream: on this toolchain,
    // std::ofstream construction intermittently segfaults after a large
    // run (confirmed via bisection to be specific to libstdc++'s ofstream
    // machinery on this MinGW build, not a bug in the simulation itself
    // -- plain fopen/fprintf and generic heap allocation both run
    // reliably in the same position). Documented here rather than
    // silently worked around.
    FILE* csv = std::fopen("simulation_results.csv", "w");
    if (csv == nullptr) {
        std::cerr << "Failed to open simulation_results.csv for writing\n";
        return 1;
    }
    std::fprintf(csv, "step,mid_price,mm_bid,mm_ask,mm_inventory,mm_realized_pnl,mm_mark_to_market_pnl\n");
    for (const auto& r : results) {
        std::fprintf(csv, "%d,%.4f,%.4f,%.4f,%lld,%.4f,%.4f\n",
                      r.step, r.mid_price,
                      lob::to_display_price(r.mm_bid), lob::to_display_price(r.mm_ask),
                      static_cast<long long>(r.mm_inventory),
                      r.mm_realized_pnl, r.mm_mark_to_market_pnl);
    }
    std::fclose(csv);

    const auto& last = results.back();
    std::cout << "Simulation complete: " << results.size() << " steps\n";
    std::cout << std::fixed;
    std::printf("Final mid price:        %.4f\n", last.mid_price);
    std::printf("Final inventory:        %lld\n", static_cast<long long>(last.mm_inventory));
    std::printf("Final realized P&L:     %.4f\n", last.mm_realized_pnl);
    std::printf("Final mark-to-mkt P&L:  %.4f\n", last.mm_mark_to_market_pnl);
    std::cout << "Wrote simulation_results.csv\n";

    return 0;
}
