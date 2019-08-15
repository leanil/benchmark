cmake_minimum_required (VERSION 3.12)

project (lambdagen_benchmark LANGUAGES CXX ASM)

option(BENCHMARK_ENABLE_GTEST_TESTS "Enable building the unit tests which depend on gtest" OFF)
add_subdirectory(extern/benchmark)

add_subdirectory(src)
add_subdirectory(app)
