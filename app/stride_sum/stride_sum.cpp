#include "data.h"
#include "sizes.h"
#include "util.h"
#include <benchmark/benchmark.h>
#include <x86intrin.h>

using namespace std;

double stride_sum_1vec(double *A, int s, int n, int rep)
{
    volatile double ret = 0;
    while (rep--)
    {
        __m128d sum = _mm_setzero_pd();
        for (int i = 0; i < n; i += s)
        {
            sum = _mm_add_pd(sum, _mm_load_pd(A + i));
        }
        __m128d tmp = _mm_unpackhi_pd(sum, _mm_setzero_pd());
        sum = _mm_add_pd(sum, tmp);
        ret = _mm_cvtsd_f64(sum);
    }
    return ret;
}

double stride_sum_6vec(double *A, int s, int n, int rep)
{
    volatile double ret = 0;
    while(rep--)
    {
        double *B = A;
        __m128d sum0, sum1, sum2, sum3, sum4, sum5;
        sum0 = sum1 = sum2 = sum3 = sum4 = sum5 = _mm_setzero_pd();
        for (int i = 0; i < n; i += 6, B += 6 * s)
        {
            sum0 = _mm_add_pd(sum0, _mm_load_pd(B + 0 * s));
            sum1 = _mm_add_pd(sum1, _mm_load_pd(B + 1 * s));
            sum2 = _mm_add_pd(sum2, _mm_load_pd(B + 2 * s));
            sum3 = _mm_add_pd(sum3, _mm_load_pd(B + 3 * s));
            sum4 = _mm_add_pd(sum4, _mm_load_pd(B + 4 * s));
            sum5 = _mm_add_pd(sum5, _mm_load_pd(B + 5 * s));
        }
        sum0 = _mm_add_pd(sum0, sum1);
        sum0 = _mm_add_pd(sum0, sum2);
        sum0 = _mm_add_pd(sum0, sum3);
        sum0 = _mm_add_pd(sum0, sum4);
        sum0 = _mm_add_pd(sum0, sum5);
        sum1 = _mm_unpackhi_pd(sum0, sum1);
        sum0 = _mm_add_pd(sum0, sum1);
        ret = _mm_cvtsd_f64(sum0);
    }
    return ret;
}

template <int BATCH_1, int BATCH_2>
void stride_sum_args(benchmark::internal::Benchmark *bench)
{
    Data<double>::max_size(0, Sizes::set_mat(bench, BATCH_1, BATCH_2));
}

template <typename F>
void stride_sum(benchmark::State &state, F f)
{
    double *data = Data<double>::get();
    while (state.KeepRunningBatch(state.max_iterations))
        benchmark::DoNotOptimize(f(data, state.range(0), state.range(1), state.max_iterations));
    set_proc_speed(state, state.range(1) * 2 * sizeof(double));
}

BENCHMARK_CAPTURE(stride_sum, 1vec, stride_sum_1vec)->Apply(stride_sum_args<2, 1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(stride_sum, 6vec, stride_sum_6vec)->Apply(stride_sum_args<2, 6>)->ComputeStatistics("max", max_stats);

BENCHMARK_MAIN();
