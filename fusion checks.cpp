#include "mkl.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using namespace std;

using real = float;

auto random_vector(int size) {
    vector<real> vec(size);
    static default_random_engine gen;
    uniform_real_distribution<real> dist;
    generate(vec.begin(), vec.end(), [&]() {return dist(gen); });
    return vec;
}

void baseline_sum(int size, int test_cnt) {
    cout << "baseline_sum (" << size << "): ";
    vector<int> times(test_cnt);
    auto A = random_vector(size);
    for (int t = 0; t < test_cnt; ++t) {
        auto from = chrono::high_resolution_clock::now();
        cblas_sasum(size, A.data(), 1);
        auto to = chrono::high_resolution_clock::now();
        times[t] = (int)chrono::duration_cast<chrono::milliseconds>(to - from).count();
    }
    cout << "min: " << *min_element(times.begin(), times.end());
    cout << " avg: " << (double)accumulate(times.begin(), times.end(), 0) / test_cnt << endl;
    for (int t : times)
        cout << t << " ";
    cout << endl;
}

int main()
{
    for (int i = 20; i <= 28; ++i) {
        baseline_sum((1 << i) - 1, 100);
        baseline_sum(1 << i, 100);
    }
    system("PAUSE");
    return 0;
}
