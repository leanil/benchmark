#include <array>
#include <benchmark/benchmark.h>
#include "data.h"
#include "sizes.h"
#include "util.h"

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

constexpr array bench_funs{bench0, bench1, bench2, bench3, bench4, bench5};

void bench(benchmark::State &state)
{
    double *data = Data<double>::get();
    for (auto _ : state)
        bench_funs[state.range(0)](data, state.range(1));
    set_proc_speed(state, state.range(1) * 8);
}
BENCHMARK(bench)->Apply(Sizes<double>::set<bench_funs.size()>)->ComputeStatistics("max", max_stats);

extern "C"
{
    double bench6(double *A, int S, int rep);
    double bench7(double *A, int S, int rep);
    double bench8(double *A, int S, int rep);
    double bench9(double *A, int S, int rep);
}

constexpr array asm_funs{bench6,bench7,bench8,bench9};
void asm_bench(benchmark::State &state)
{
    double *data = Data<double>::get();
    while (state.KeepRunningBatch(state.max_iterations))
        asm_funs[state.range(0)](data, state.range(1), state.max_iterations);
    set_proc_speed(state, state.range(1) * 8);
}
BENCHMARK(asm_bench)->Apply(Sizes<double>::set<asm_funs.size()>)->ComputeStatistics("max", max_stats);

BENCHMARK_MAIN();