#pragma once
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using scl_t = float;
using time_precision = std::chrono::microseconds;

void clear_cache();

struct TestResult {
    std::vector<int> times;
    int min_time, max_time;
    double avg_time;
    bool fail = false;
};

template<typename Benchmark>
TestResult test_helper(int size, std::vector<scl_t*> const& data, int test_cnt) {
    return tester(Benchmark(size, data), test_cnt);
}

template<typename Expression>
TestResult tester(Expression expr, int test_cnt) {
    TestResult result;
    result.times.reserve(test_cnt);
    for (int t = 0; t < test_cnt; ++t) {
        clear_cache();
        auto from = std::chrono::high_resolution_clock::now();
        auto x = expr.eval();
        auto to = std::chrono::high_resolution_clock::now();
        if (!expr.check(x)) {
            result.fail = true;
            return result;
        }
        result.times.push_back((int)std::chrono::duration_cast<time_precision>(to - from).count());
    }
    result.min_time = *std::min_element(result.times.begin(), result.times.end());
    result.max_time = *std::max_element(result.times.begin(), result.times.end());
    result.avg_time = (double)std::accumulate(result.times.begin(), result.times.end(), 0) / test_cnt;
    return result;
}

struct Histogram {
    int from, to, bin_cnt;
    double bin_size;
    std::vector<int> freq;
};

Histogram make_histogram(TestResult const& result, int from, int to, int bin_cnt);

Histogram make_histogram(TestResult const& result, int bin_cnt);

std::ostream& operator<<(std::ostream& out, Histogram const& hist);

scl_t random_scalar();

class AlignedData {
public:
    AlignedData(size_t size, size_t alignment);
    //~AlignedData();
    scl_t* data;
private:
    void* orig;
    static std::default_random_engine gen;
};

std::vector<scl_t> random_vector(int size);