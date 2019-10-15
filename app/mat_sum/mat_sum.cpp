#include "data.h"
#include "sizes.h"
#include "util.h"
#include <benchmark/benchmark.h>
#include <x86intrin.h>

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

double mat_sum_6(double *A, int n, int m)
{
    __m128d sum0, sum1, sum2, sum3, sum4, sum5;
    sum0 = sum1 = sum2 = sum3 = sum4 = sum5 = _mm_setzero_pd();
    for (int j = 0; j < m; j += 2)
        for (int i = 0; i < n; i += 6)
        {
            sum0 = _mm_add_pd(sum0, _mm_load_pd(A + i * m + j));
            sum1 = _mm_add_pd(sum1, _mm_load_pd(A + (i + 1) * m + j));
            sum2 = _mm_add_pd(sum2, _mm_load_pd(A + (i + 2) * m + j));
            sum3 = _mm_add_pd(sum3, _mm_load_pd(A + (i + 3) * m + j));
            sum4 = _mm_add_pd(sum4, _mm_load_pd(A + (i + 4) * m + j));
            sum5 = _mm_add_pd(sum5, _mm_load_pd(A + (i + 5) * m + j));
        }
    sum0 = _mm_add_pd(sum0, sum1);
    sum0 = _mm_add_pd(sum0, sum2);
    sum0 = _mm_add_pd(sum0, sum3);
    sum0 = _mm_add_pd(sum0, sum4);
    sum0 = _mm_add_pd(sum0, sum5);
    sum1 = _mm_unpackhi_pd(sum0, sum1);
    sum0 = _mm_add_pd(sum0, sum1);
    return _mm_cvtsd_f64(sum0);
}

template <int BATCH_0, int BATCH_1>
void stride_sum_args(benchmark::internal::Benchmark *bench)
{
    Data<double>::max_size(0, Sizes::set_mat(bench, BATCH_0, BATCH_1));
}

template <typename F>
void stride_sum(benchmark::State &state, F f)
{
    double *data = Data<double>::get();
    for (auto _ : state)
        benchmark::DoNotOptimize(f(data, state.range(0), state.range(1)));
    set_proc_speed(state, state.range(0) * state.range(1) * sizeof(double));
}

BENCHMARK_CAPTURE(stride_sum, 1var, mat_sum_0)->Apply(stride_sum_args<1, 1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 2var, mat_sum_1)->Apply(stride_sum_args<2, 1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 3var, mat_sum_2)->Apply(stride_sum_args<3, 1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 6var, mat_sum_3)->Apply(stride_sum_args<6, 1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 6vec, mat_sum_6)->Apply(stride_sum_args<6, 2>)->ComputeStatistics("max", max_stats);

extern "C"
{
    double mat_sum_5(double *a, int n, int m, int rep);
}

template <typename F>
void stride_sum_asm(benchmark::State &state, F f)
{
    double *A = Data<double>::get();
    while (state.KeepRunningBatch(state.max_iterations))
        f(A, state.range(0), state.range(1), state.max_iterations);
    set_proc_speed(state, state.range(0) * state.range(1) * sizeof(double));
}
BENCHMARK_CAPTURE(stride_sum_asm, 6var, mat_sum_5)->Apply(stride_sum_args<6, 2>)->ComputeStatistics("max", max_stats);

BENCHMARK_MAIN();
