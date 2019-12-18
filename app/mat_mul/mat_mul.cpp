#include "mat_mul.hpp"
#include <benchmark/benchmark.h>
#include <numeric>
#include "data.h"
#include "sizes.h"
#include "util.h"

using namespace std;

template <int TILE_LCM>
void mat_mul_args(benchmark::internal::Benchmark *bench) {
    int mx = Sizes::set(bench, TILE_LCM);
    Data<double>::max_size(0, mx * mx);
    Data<double>::max_size(1, mx * mx);
    Data<double>::max_size(2, mx * mx);
}

template <typename F>
void mat_mul_square(benchmark::State &state, F f) {
    double *A = Data<double>::get(0), *B = Data<double>::get(1), *C = Data<double>::get(2);
    for (auto _ : state) {
        f(A, B, C, state.range(0), state.range(0), state.range(0));
        benchmark::ClobberMemory();
    }
    set_proc_speed(state, state.range(0) * state.range(0) * state.range(0));
}

template <typename F>
void mat_mul_outer(benchmark::State &state, F f, int tile_lcm) {
    double *A = Data<double>::get(0), *B = Data<double>::get(1), *C = Data<double>::get(2);
    for (auto _ : state) {
        f(A, B, C, state.range(0), state.range(0), tile_lcm);
        benchmark::ClobberMemory();
    }
    set_proc_speed(state, state.range(0) * state.range(0) * tile_lcm);
}

template <typename F>
void mat_mul_inner(benchmark::State &state, F f, int tile_lcm) {
    double *A = Data<double>::get(0), *B = Data<double>::get(1), *C = Data<double>::get(2);
    for (auto _ : state) {
        f(A, B, C, tile_lcm, tile_lcm, state.range(0));
        benchmark::ClobberMemory();
    }
    set_proc_speed(state, tile_lcm * tile_lcm * state.range(0));
}

BENCHMARK_CAPTURE(mat_mul_square, naive, mat_mul_0)->Apply(mat_mul_args<1>);
BENCHMARK_CAPTURE(mat_mul_outer, naive, mat_mul_0, 1)->Apply(mat_mul_args<1>);
BENCHMARK_CAPTURE(mat_mul_inner, naive, mat_mul_0, 1)->Apply(mat_mul_args<1>);
BENCHMARK_CAPTURE(mat_mul_square, tiled, mat_mul_1)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_outer, tiled, mat_mul_1, 4)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_inner, tiled, mat_mul_1, 4)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_square, tiled_2_4, mat_mul_2<2, 4>)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_outer, tiled_2_4, mat_mul_2<2, 4>, 4)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_inner, tiled_2_4, mat_mul_2<2, 4>, 4)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_square, tiled_4_4, mat_mul_2<4, 4>)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_outer, tiled_4_4, mat_mul_2<4, 4>, 4)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_inner, tiled_4_4, mat_mul_2<4, 4>, 4)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_square, tiled_2_8, mat_mul_2<2, 8>)->Apply(mat_mul_args<8>);
BENCHMARK_CAPTURE(mat_mul_outer, tiled_2_8, mat_mul_2<2, 8>, 8)->Apply(mat_mul_args<8>);
BENCHMARK_CAPTURE(mat_mul_inner, tiled_2_8, mat_mul_2<2, 8>, 8)->Apply(mat_mul_args<8>);
BENCHMARK_CAPTURE(mat_mul_square, tiled_8_8, mat_mul_2<8, 8>)->Apply(mat_mul_args<8>);
BENCHMARK_CAPTURE(mat_mul_outer, tiled_8_8, mat_mul_2<8, 8>, 8)->Apply(mat_mul_args<8>);
BENCHMARK_CAPTURE(mat_mul_inner, tiled_8_8, mat_mul_2<8, 8>, 8)->Apply(mat_mul_args<8>);
BENCHMARK_CAPTURE(mat_mul_square, eigen, eigen_map)->Apply(mat_mul_args<1>);
BENCHMARK_CAPTURE(mat_mul_outer, eigen, eigen_map, 1)->Apply(mat_mul_args<1>);
BENCHMARK_CAPTURE(mat_mul_inner, eigen, eigen_map, 1)->Apply(mat_mul_args<1>);
BENCHMARK_CAPTURE(mat_mul_square, eigen2, eigen_map)->Apply(mat_mul_args<8>);
BENCHMARK_CAPTURE(mat_mul_outer, eigen2, eigen_map, 8)->Apply(mat_mul_args<8>);
BENCHMARK_CAPTURE(mat_mul_inner, eigen2, eigen_map, 8)->Apply(mat_mul_args<8>);
BENCHMARK_CAPTURE(mat_mul_square, l1__reg_tiled_36_4, mat_mul_3<36,36,36, 4,4>)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_outer, l1_reg_tiled_36_4, mat_mul_3<36, 36, 36, 4, 4>, 4)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_inner, l1_reg_tiled_36_4, mat_mul_3<36, 36, 36, 4, 4>, 4)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_square, l1__reg_tiled_16_4, mat_mul_3<16, 16, 16, 4, 4>)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_outer, l1_reg_tiled_16_4, mat_mul_3<16, 16, 16, 4, 4>, 4)->Apply(mat_mul_args<4>);
BENCHMARK_CAPTURE(mat_mul_inner, l1_reg_tiled_16_4, mat_mul_3<16, 16, 16, 4, 4>, 4)->Apply(mat_mul_args<4>);
BENCHMARK_MAIN();

/*
double mat_mul_1(double *A, double *B, int r, int c)
{
    double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < r; ++j)
            for (int k = 0; k < c; k += 6)
            {
                sum0 += A[i * c + k] * B[k * r + j];
                sum1 += A[i * c + k + 1] * B[(k + 1) * r + j];
                sum2 += A[i * c + k + 2] * B[(k + 2) * r + j];
                sum3 += A[i * c + k + 3] * B[(k + 3) * r + j];
                sum4 += A[i * c + k + 4] * B[(k + 4) * r + j];
                sum5 += A[i * c + k + 5] * B[(k + 5) * r + j];
            }
    return sum0 + sum1 + sum2 + sum3 + sum4 + sum5;
}

double mat_mul_2(double *A, double *B, int r, int c)
{
    double sum = 0;
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < r; ++j)
            for (int k = 0; k < c; ++k)
                sum += A[i * c + k] * B[j * c + k];
    return sum;
}
*/