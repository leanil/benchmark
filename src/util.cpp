#include "util.h"
#include <algorithm>

using namespace std;

void set_proc_speed(benchmark::State &state, int bytes)
{
    state.counters["data_size"] = bytes;
    state.counters["processing_speed"] = benchmark::Counter(
        bytes, benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1024);
}

double min_stats(const vector<double> &v)
{
    return *min_element(v.begin(), v.end());
}
double max_stats(const vector<double> &v)
{
    return *max_element(v.begin(), v.end());
}
