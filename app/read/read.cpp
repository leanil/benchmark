#include "data.h"
#include "sizes.h"
#include "util.h"

using namespace std;

extern "C"
{
    void bench1(char *, size_t, size_t);
    void bench2(char *, size_t, size_t);
    void bench3(char *, size_t, size_t);
    void bench4(char *, size_t, size_t);
}

template <int BATCH>
void read_args(benchmark::internal::Benchmark *bench)
{
    Data<char>::max_size(0, Sizes::set(bench, BATCH));
}

template<typename F>
void read(benchmark::State &state, F f)
{
    char *data = Data<char>::get();
    while (state.KeepRunningBatch(state.max_iterations))
        f(data, state.range(0), state.max_iterations);
    set_proc_speed(state, state.range(0) * sizeof(char));
}

BENCHMARK_CAPTURE(read, movdqa, bench1)->Apply(read_args<256>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(read, vmovdqa, bench2)->Apply(read_args<256>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(read, movapd, bench2)->Apply(read_args<256>)->ComputeStatistics("max", max_stats);
BENCHMARK_CAPTURE(read, vmovapd, bench3)->Apply(read_args<256>)->ComputeStatistics("max", max_stats);
BENCHMARK_MAIN();
