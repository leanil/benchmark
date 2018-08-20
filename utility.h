#pragma once
#include <vector>

using scl_t = float;

struct TestResult {
    std::vector<int> times;
    int min_time;
    double avg_time;
    bool fail = false;
};

template<typename Expression>
TestResult tester(Expression expr, int size, int test_cnt) {
    TestResult result;
    result.times.reserve(test_cnt);
    expr.init(size);
    for (int t = 0; t < test_cnt; ++t) {
        auto from = chrono::high_resolution_clock::now();
        auto x = expr.eval();
        auto to = chrono::high_resolution_clock::now();
        if (!expr.check(x)) {
            result.fail = true;
            return result;
        }
        result.times.push_back((int)chrono::duration_cast<chrono::milliseconds>(to - from).count());
    }
    result.min_time = *min_element(result.times.begin(), result.times.end());
    result.avg_time = (double)accumulate(result.times.begin(), result.times.end(), 0) / test_cnt;
    return result;
}

scl_t random_scalar();