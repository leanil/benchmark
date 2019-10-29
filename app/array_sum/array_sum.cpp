#include "data.h"
#include "sizes.h"
#include "util.h"
#include <benchmark/benchmark.h>
#include <x86intrin.h>

using namespace std;

double array_sum_1var(double *A, int S)
{
    double sum = 0;
    for (int i = 0; i < S; ++i)
        sum += A[i];
    return sum;
}

double array_sum_2var(double *A, int S)
{
    double sum0 = 0, sum1 = 0;
    for (int i = 0; i < S; i += 2)
    {
        sum0 += A[i];
        sum1 += A[i + 1];
    }
    return sum0 + sum1;
}

double array_sum_3var(double *A, int S)
{
    double sum0 = 0, sum1 = 0, sum2 = 0;
    for (int i = 0; i < S; i += 3)
    {
        sum0 += A[i];
        sum1 += A[i + 1];
        sum2 += A[i + 2];
    }
    return sum0 + sum1 + sum2;
}

double array_sum_4var(double *A, int S)
{
    double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    for (int i = 0; i < S; i += 4)
    {
        sum0 += A[i];
        sum1 += A[i + 1];
        sum2 += A[i + 2];
        sum3 += A[i + 3];
    }
    return sum0 + sum1 + sum2 + sum3;
}

double array_sum_6var(double *A, int S)
{
    double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
    for (int i = 0; i < S; i += 6)
    {
        sum0 += A[i];
        sum1 += A[i + 1];
        sum2 += A[i + 2];
        sum3 += A[i + 3];
        sum4 += A[i + 4];
        sum5 += A[i + 5];
    }
    return sum0 + sum1 + sum2 + sum3 + sum4 + sum5;
}

double array_sum_12var(double *A, int S)
{
    double sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0, sum9 = 0, sum10 = 0, sum11 = 0;
    for (int i = 0; i < S; i += 12)
    {
        sum0 += A[i];
        sum1 += A[i + 1];
        sum2 += A[i + 2];
        sum3 += A[i + 3];
        sum4 += A[i + 4];
        sum5 += A[i + 5];
        sum6 += A[i + 6];
        sum7 += A[i + 7];
        sum8 += A[i + 8];
        sum9 += A[i + 9];
        sum10 += A[i + 10];
        sum11 += A[i + 11];
    }
    return sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8 + sum9 + sum10 + sum11;
}

double array_sum_1vec(double *A, int S)
{
    __m128d sum = _mm_setzero_pd();
    for (int i = 0; i < S; i += 2)
    {
        sum = _mm_add_pd(sum, _mm_load_pd(A + i));
    }
    __m128d tmp = _mm_unpackhi_pd(sum, _mm_setzero_pd());
    sum = _mm_add_pd(sum, tmp);
    return _mm_cvtsd_f64(sum);
}

double array_sum_6vec(double *A, int S)
{
    __m128d sum0, sum1, sum2, sum3, sum4, sum5;
    sum0 = sum1 = sum2 = sum3 = sum4 = sum5 = _mm_setzero_pd();
    for (int i = 0; i < S; i += 12)
    {
        sum0 = _mm_add_pd(sum0, _mm_load_pd(A + i));
        sum1 = _mm_add_pd(sum1, _mm_load_pd(A + i + 2));
        sum2 = _mm_add_pd(sum2, _mm_load_pd(A + i + 4));
        sum3 = _mm_add_pd(sum3, _mm_load_pd(A + i + 6));
        sum4 = _mm_add_pd(sum4, _mm_load_pd(A + i + 8));
        sum5 = _mm_add_pd(sum5, _mm_load_pd(A + i + 10));
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

extern "C"
{
    double array_sum_asm_nomov(double *A, int S);
    double array_sum_asm_mov(double *A, int S);
}

template <int BATCH>
void seq_sum_args(benchmark::internal::Benchmark *bench)
{
    Data<double>::max_size(0, Sizes::set(bench, BATCH));
}

template <typename F>
void seq_sum(benchmark::State &state, F f)
{
    double *data = Data<double>::get();
    for (auto _ : state)
        benchmark::DoNotOptimize(f(data, state.range(0)));
    set_proc_speed(state, state.range(0) * sizeof(double));
}

BENCHMARK_CAPTURE(seq_sum, 1var, array_sum_1var)->Apply(seq_sum_args<1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, 2var, array_sum_2var)->Apply(seq_sum_args<2>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, 3var, array_sum_3var)->Apply(seq_sum_args<3>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, 4var, array_sum_4var)->Apply(seq_sum_args<4>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, 6var, array_sum_6var)->Apply(seq_sum_args<6>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, 12var, array_sum_12var)->Apply(seq_sum_args<12>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, 1vec, array_sum_1vec)->Apply(seq_sum_args<2>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, 6vec, array_sum_6vec)->Apply(seq_sum_args<12>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, asm_nomov, array_sum_asm_nomov)->Apply(seq_sum_args<12>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, asm_mov, array_sum_asm_mov)->Apply(seq_sum_args<12>)->ComputeStatistics("max", max_stats);

BENCHMARK_MAIN();
