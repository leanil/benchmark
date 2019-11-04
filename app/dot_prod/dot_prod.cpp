#include "data.h"
#include "sizes.h"
#include "util.h"
#include <benchmark/benchmark.h>
#include <x86intrin.h>

using namespace std;

double dot_prod_1var(double *A, double *B, int S, int rep)
{
    volatile double ret = 0;
    while (rep--)
    {
        double sum = 0;
        for (int i = 0; i < S; ++i)
            sum += A[i] * B[i];
        ret = sum;
    }
    return ret;
}

double dot_prod_12var(double *A, double *B, int S, int rep)
{
    volatile double ret = 0;
    while (rep--)
    {
        double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0, sum10 = 0, sum11 = 0;
        for (int i = 0; i < S; i += 12)
        {
            sum0 += A[i] * B[i];
            sum1 += A[i + 1] * B[i + 1];
            sum2 += A[i + 2] * B[i + 2];
            sum3 += A[i + 3] * B[i + 3];
            sum4 += A[i + 4] * B[i + 4];
            sum5 += A[i + 5] * B[i + 5];
            sum6 += A[i + 6] * B[i + 6];
            sum7 += A[i + 7] * B[i + 7];
            sum8 += A[i + 8] * B[i + 8];
            sum9 += A[i + 9] * B[i + 9];
            sum10 += A[i + 10] * B[i + 10];
            sum11 += A[i + 11] * B[i + 11];
        }
        ret = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8 + sum9 + sum10 + sum11;
    }
    return ret;
}

double dot_prod_3vec(double *A, double *B, int S, int rep)
{
    volatile double ret = 0;
    while (rep--)
    {
        __m128d sum0, sum1, sum2;
        sum0 = sum1 = sum2 = _mm_setzero_pd();
        for (int i = 0; i < S; i += 6)
        {
            sum0 = _mm_add_pd(sum0, _mm_mul_pd(_mm_load_pd(A + i), _mm_load_pd(B + i)));
            sum1 = _mm_add_pd(sum1, _mm_mul_pd(_mm_load_pd(A + i + 2), _mm_load_pd(B + i + 2)));
            sum2 = _mm_add_pd(sum2, _mm_mul_pd(_mm_load_pd(A + i + 4), _mm_load_pd(B + i + 4)));
        }
        sum0 = _mm_add_pd(sum0, sum1);
        sum0 = _mm_add_pd(sum0, sum2);
        sum1 = _mm_unpackhi_pd(sum0, sum1);
        sum0 = _mm_add_pd(sum0, sum1);
        ret = _mm_cvtsd_f64(sum0);
    }
    return ret;
}

double dot_prod_6vec(double *A, double *B, int S, int rep)
{
    volatile double ret = 0;
    while (rep--)
    {
        __m128d sum0, sum1, sum2, sum3, sum4, sum5;
        sum0 = sum1 = sum2 = sum3 = sum4 = sum5 = _mm_setzero_pd();
        for (int i = 0; i < S; i += 12)
        {
            sum0 = _mm_add_pd(sum0, _mm_mul_pd(_mm_load_pd(A + i), _mm_load_pd(B + i)));
            sum1 = _mm_add_pd(sum1, _mm_mul_pd(_mm_load_pd(A + i + 2), _mm_load_pd(B + i + 2)));
            sum2 = _mm_add_pd(sum2, _mm_mul_pd(_mm_load_pd(A + i + 4), _mm_load_pd(B + i + 4)));
            sum3 = _mm_add_pd(sum3, _mm_mul_pd(_mm_load_pd(A + i + 6), _mm_load_pd(B + i + 6)));
            sum4 = _mm_add_pd(sum4, _mm_mul_pd(_mm_load_pd(A + i + 8), _mm_load_pd(B + i + 8)));
            sum5 = _mm_add_pd(sum5, _mm_mul_pd(_mm_load_pd(A + i + 10), _mm_load_pd(B + i + 10)));
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

extern "C"
{
    double bench6(double *A, double *B, int S, int rep);
}

template <int BATCH>
void dot_prod_args(benchmark::internal::Benchmark *bench)
{
    int mx = Sizes::set(bench, BATCH);
    Data<double>::max_size(0, mx);
    Data<double>::max_size(1, mx);
}

template <typename F>
void dot_prod(benchmark::State &state, F f)
{
    double *A = Data<double>::get(0), *B = Data<double>::get(1);
    while (state.KeepRunningBatch(state.max_iterations))
        benchmark::DoNotOptimize(f(A, B, state.range(0), state.max_iterations));
    set_proc_speed(state, 2 * state.range(0) * sizeof(double));
}

BENCHMARK_CAPTURE(dot_prod, 1var, dot_prod_1var)->Apply(dot_prod_args<1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(dot_prod, 12var, dot_prod_12var)->Apply(dot_prod_args<12>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(dot_prod, 3vec, dot_prod_3vec)->Apply(dot_prod_args<6>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(dot_prod, 6vec, dot_prod_6vec)->Apply(dot_prod_args<12>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(dot_prod, asm, bench6)->Apply(dot_prod_args<12>)->ComputeStatistics("max", max_stats);

BENCHMARK_MAIN();
