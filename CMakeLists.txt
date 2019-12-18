cmake_minimum_required (VERSION 3.12)

project (lambdagen_benchmark LANGUAGES CXX ASM)

option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
add_subdirectory(extern/benchmark)

enable_testing()
add_subdirectory(src)
add_subdirectory(app)
