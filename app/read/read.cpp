
#include "data.h"
#include "sizes.h"
#include "util.h"
#include <array>

using namespace std;

extern "C"
{
    void bench1(char *, size_t, size_t);
    void bench2(char *, size_t, size_t);
    void bench3(char *, size_t, size_t);
    void bench4(char *, size_t, size_t);
}

constexpr array bench_funs{bench1, bench2, bench3, bench4};

void read(benchmark::State &state)
{
    char *data = Data<char>::get();
    while (state.KeepRunningBatch(state.max_iterations))
        bench_funs[state.range(0)](data, state.range(1), state.max_iterations);
    set_proc_speed(state, state.range(1));
}
BENCHMARK(read)->Apply(Sizes<char>::set<bench_funs.size(),256>)->ComputeStatistics("max", max_stats);

BENCHMARK_MAIN();
