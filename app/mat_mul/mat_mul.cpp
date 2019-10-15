#include "data.h"
#include "sizes.h"
#include "util.h"
#include <benchmark/benchmark.h>

using namespace std;

double mat_mul_0(double *A, double *B, int r, int c)
{
    double sum = 0;
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < r; ++j)
            for (int k = 0; k < c; ++k)
                sum += A[i * c + k] * B[k * r + j];
    return sum;
}

template <int BATCH_0, int BATCH_1>
void mat_mul_args(benchmark::internal::Benchmark *bench)
{
    int mx = Sizes::set_mat(bench, BATCH_0, BATCH_1);
    Data<double>::max_size(0, mx);
    Data<double>::max_size(1, mx);
}

template <typename F>
void mat_mul(benchmark::State &state, F f)
{
    double *A = Data<double>::get(0), *B = Data<double>::get(1);
    for (auto _ : state)
        benchmark::DoNotOptimize(f(A, B, state.range(0), state.range(1)));
    set_proc_speed(state, 2 * state.range(0) * state.range(1) * sizeof(double));
}

BENCHMARK_CAPTURE(mat_mul, naive, mat_mul_0)->Apply(mat_mul_args<1,1>)->ComputeStatistics("max", max_stats);

BENCHMARK_MAIN();
