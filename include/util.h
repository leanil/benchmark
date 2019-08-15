#ifndef UTIL_H_
#define UTIL_H_

#include <vector>
#include <benchmark/benchmark.h>

void set_proc_speed(benchmark::State &state, int bytes);

double min_stats(const std::vector<double> &v);
double max_stats(const std::vector<double> &v);

#endif // UTIL_H_
