#include "data.h"
#include "sizes.h"
#include "util.h"
#include <array>
#include <benchmark/benchmark.h>

using namespace std;

double mat_sum_0(double *a, int n, int m)
{
    double sum = 0;
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < n; ++i)
            sum += a[i * m + j];
    return sum;
}

double mat_sum_1(double *a, int n, int m)
{
    double sum0 = 0, sum1 = 0;
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < n; i += 2)
        {
            sum0 += a[i * m + j];
            sum1 += a[(i + 1) * m + j];
        }
    return sum0 + sum1;
}

double mat_sum_2(double *a, int n, int m)
{
    double sum0 = 0, sum1 = 0, sum2 = 0;
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < n; i += 3)
        {
            sum0 += a[i * m + j];
            sum1 += a[(i + 1) * m + j];
            sum2 += a[(i + 2) * m + j];
        }
    return sum0 + sum1 + sum2;
}

double mat_sum_3(double *a, int n, int m)
{
    double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < n; i += 6)
        {
            sum0 += a[i * m + j];
            sum1 += a[(i + 1) * m + j];
            sum2 += a[(i + 2) * m + j];
            sum3 += a[(i + 3) * m + j];
            sum4 += a[(i + 4) * m + j];
            sum5 += a[(i + 5) * m + j];
        }
    return sum0 + sum1 + sum2 + sum3 + sum4 + sum5;
}

double mat_sum_4(double *a, int n, int m)
{
    double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0, sum10 = 0, sum11 = 0;
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < n; i += 12)
        {
            sum0 += a[i * m + j];
            sum1 += a[(i + 1) * m + j];
            sum2 += a[(i + 2) * m + j];
            sum3 += a[(i + 3) * m + j];
            sum4 += a[(i + 4) * m + j];
            sum5 += a[(i + 5) * m + j];
            sum6 += a[(i + 6) * m + j];
            sum7 += a[(i + 7) * m + j];
            sum8 += a[(i + 8) * m + j];
            sum9 += a[(i + 9) * m + j];
            sum10 += a[(i + 10) * m + j];
            sum11 += a[(i + 11) * m + j];
        }
    return sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8 + sum9 + sum10 + sum11;
}

template <typename F>
void stride_sum(benchmark::State &state, F f, int rows, int cols)
{
    double *data = Data<double>::get();
    for (auto _ : state)
        benchmark::DoNotOptimize(f(data, state.range(rows), state.range(cols)));
    set_proc_speed(state, state.range(0) * state.range(1) * sizeof(double));
    state.counters["x_label:dimension size (Bytes)"] = state.range(1) * sizeof(double);
}
BENCHMARK_CAPTURE(stride_sum, 1var_rows, mat_sum_0, 0, 1)->Apply(Sizes<double>::set_mat<1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 2var_rows, mat_sum_1, 0, 1)->Apply(Sizes<double>::set_mat<1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 3var_rows, mat_sum_2, 0, 1)->Apply(Sizes<double>::set_mat<1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 6var_rows, mat_sum_3, 0, 1)->Apply(Sizes<double>::set_mat<1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 12var_rows, mat_sum_4, 0, 1)->Apply(Sizes<double>::set_mat<1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 1var_cols, mat_sum_0, 1, 0)->Apply(Sizes<double>::set_mat<1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 2var_cols, mat_sum_1, 1, 0)->Apply(Sizes<double>::set_mat<2>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 3var_cols, mat_sum_2, 1, 0)->Apply(Sizes<double>::set_mat<3>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 6var_cols, mat_sum_3, 1, 0)->Apply(Sizes<double>::set_mat<6>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 12var_cols, mat_sum_4, 1, 0)->Apply(Sizes<double>::set_mat<12>)->ComputeStatistics("max", max_stats);

BENCHMARK_MAIN();
