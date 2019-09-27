#include "data.h"
#include "sizes.h"
#include "util.h"
#include <benchmark/benchmark.h>

using namespace std;

double bench0(double *A, int S)
{
    double sum = 0;
    for (int i = 0; i < S; ++i)
        sum += A[i];
    return sum;
}

double bench1(double *A, int S)
{
    double sum0 = 0, sum1 = 0;
    for (int i = 0; i < S; i += 2)
    {
        sum0 += A[i];
        sum1 += A[i + 1];
    }
    return sum0 + sum1;
}

double bench2(double *A, int S)
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

double bench3(double *A, int S)
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

double bench4(double *A, int S)
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

double bench5(double *A, int S)
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

template <typename F>
void seq_sum(benchmark::State &state, F f)
{
    double *data = Data<double>::get();
    for (auto _ : state)
        benchmark::DoNotOptimize(f(data, state.range(0)));
    set_proc_speed(state, state.range(0) * sizeof(double));
    state.counters["x_label:data size (Bytes)"] = state.range(0) * sizeof(double);
}
BENCHMARK_CAPTURE(seq_sum, 1var, bench0)->Apply(Sizes<double>::set<1>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, 2var, bench1)->Apply(Sizes<double>::set<2>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, 3var, bench2)->Apply(Sizes<double>::set<3>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, 4var, bench3)->Apply(Sizes<double>::set<4>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, 6var, bench4)->Apply(Sizes<double>::set<6>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum, 12var, bench5)->Apply(Sizes<double>::set<12>)->ComputeStatistics("max", max_stats);

extern "C"
{
    double bench6(double *A, int S, int rep);
    double bench7(double *A, int S, int rep);
    double bench8(double *A, int S, int rep);
    double bench9(double *A, int S, int rep);
}

template <typename F>
void seq_sum_asm(benchmark::State &state, F f)
{
    double *data = Data<double>::get();
    while (state.KeepRunningBatch(state.max_iterations))
        f(data, state.range(0), state.max_iterations);
    set_proc_speed(state, state.range(0) * sizeof(double));
    state.counters["x_label:data size (Bytes)"] = state.range(0) * sizeof(double);
}
BENCHMARK_CAPTURE(seq_sum_asm, nomov_combine, bench6)->Apply(Sizes<double>::set<12>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum_asm, mov_combine, bench7)->Apply(Sizes<double>::set<12>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum_asm, nomov_nocombine, bench8)->Apply(Sizes<double>::set<12>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(seq_sum_asm, mov_nocombine, bench9)->Apply(Sizes<double>::set<12>)->ComputeStatistics("max", max_stats);

BENCHMARK_MAIN();