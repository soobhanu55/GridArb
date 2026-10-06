cmake_minimum_required(VERSION 3.20)
project(limit_order_book LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

include(FetchContent)

# Core library: the order book, market maker, and execution simulator.
add_library(lob_core
    src/order_book.cpp
    src/market_maker.cpp
    src/execution_simulator.cpp
)
target_include_directories(lob_core PUBLIC include)

# CLI demo: runs a simulation and prints a summary.
add_executable(lob_demo src/main.cpp)
target_link_libraries(lob_demo PRIVATE lob_core)

option(LOB_BUILD_TESTS "Build unit tests" ON)
option(LOB_BUILD_BENCHMARKS "Build benchmarks" ON)

if(LOB_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        GIT_SHALLOW TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_executable(lob_tests
        tests/test_order_book.cpp
        tests/test_market_maker.cpp
    )
    target_link_libraries(lob_tests PRIVATE lob_core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(lob_tests)
endif()

if(LOB_BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
        GIT_SHALLOW TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    # Upstream benchmark's own sysinfo.cc has a non-exhaustive switch that
    # newer GCC flags as an error under -Werror; that's their code, not
    # ours, so disable -Werror for this dependency rather than patch it.
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)

    add_executable(lob_benchmarks benchmarks/bench_order_book.cpp)
    target_link_libraries(lob_benchmarks PRIVATE lob_core benchmark::benchmark)
endif()

# Dependency-free benchmark harness (see benchmarks/custom_bench.cpp for
# why this exists alongside the Google Benchmark target).
add_executable(lob_custom_bench benchmarks/custom_bench.cpp)
target_link_libraries(lob_custom_bench PRIVATE lob_core)
