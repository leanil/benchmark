#define EIGEN_DONT_PARALLELIZE
#include "Eigen/Dense"
#include "mkl.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

using namespace Eigen;
using namespace std;

using scl_t = float;
using VectorX = Matrix<scl_t, Dynamic, 1>;

auto random_vector(int size) {
    vector<scl_t> vec(size);
    static default_random_engine gen;
    uniform_real_distribution<scl_t> dist;
    generate(vec.begin(), vec.end(), [&]() {return dist(gen); });
    return vec;
}

auto random_scalar() {
    static default_random_engine gen;
    uniform_real_distribution<scl_t> dist;
    return dist(gen);
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

// a = sum_i A_i
struct BaselineSum {
    VectorX A;
    scl_t ans;
    void init(int size) {
        A = VectorX::Random(size);
        ans = eval();
    }
    scl_t eval() { return A.sum(); }
    bool check(float x) { return x == ans; }
};

// B_i = c * A_i
void baseline_prod(int size, int test_cnt) {
    cout << "baseline_prod (" << size << "): ";
    vector<int> times(test_cnt);
    VectorXf A = VectorXf::Random(size);
    auto c = random_scalar();
    for (int t = 0; t < test_cnt; ++t) {
        auto from = chrono::high_resolution_clock::now();
        (c*A).eval();
        auto to = chrono::high_resolution_clock::now();
        times[t] = (int)chrono::duration_cast<chrono::milliseconds>(to - from).count();
    }
    cout << "min: " << *min_element(times.begin(), times.end());
    cout << " avg: " << (double)accumulate(times.begin(), times.end(), 0) / test_cnt << endl;
    for (int t : times)
        cout << t << " ";
    cout << endl;
}

// sum_i A_i * B_i
void dot(int size, int test_cnt) {
    cout << "dot (" << size << "): ";
    vector<int> times(test_cnt);
    VectorXf A = VectorXf::Random(size), B = VectorXf::Random(size);
    for (int t = 0; t < test_cnt; ++t) {
        auto from = chrono::high_resolution_clock::now();
        if (A.dot(B) == -1) exit(1);
        auto to = chrono::high_resolution_clock::now();
        times[t] = (int)chrono::duration_cast<chrono::milliseconds>(to - from).count();
    }
    cout << "min: " << *min_element(times.begin(), times.end());
    cout << " avg: " << (double)accumulate(times.begin(), times.end(), 0) / test_cnt << endl;
    for (int t : times)
        cout << t << " ";
    cout << endl;
}

// sum_i A_i * B_i
struct Dot {
    VectorX A, B;
    scl_t ans;
    auto eval() { return A.dot(B); }
    void init(int size) {
        A = VectorX::Random(size);
        B = VectorX::Random(size);
        ans = eval();
    }
    bool check(float x) { return x == ans; }
};

// sum_i ( A_i + B_i ) * C_i
void dot2(int size, int test_cnt) {
    cout << "dot (" << size << "): ";
    vector<int> times(test_cnt);
    VectorXf A = VectorXf::Random(size), B = VectorXf::Random(size), C = VectorXf::Random(size);
    for (int t = 0; t < test_cnt; ++t) {
        auto from = chrono::high_resolution_clock::now();
        if((A + B).dot(C) == -1) exit(1);
        auto to = chrono::high_resolution_clock::now();
        times[t] = (int)chrono::duration_cast<chrono::milliseconds>(to - from).count();
    }
    cout << "min: " << *min_element(times.begin(), times.end());
    cout << " avg: " << (double)accumulate(times.begin(), times.end(), 0) / test_cnt << endl;
    for (int t : times)
        cout << t << " ";
    cout << endl;
}

// sum_i ( A_i + B_i ) * C_i
struct Dot2 {
    array<VectorX, 3> A;
    scl_t ans;
    auto eval() { return (A[0] + A[1]).dot(A[2]); }
    void init(int size) {
        generate(A.begin(), A.end(), [&]() {return VectorX::Random(size); });
        ans = eval();
    }
    bool check(float x) { return x == ans; }
};

struct TestResult {
    vector<int> times;
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

int main()
{
    //VectorX A(3);
    //A << 1, 10, 100;
    //cout << A.sum() << endl;
    for (int i = 25; i <= 28; ++i) {
        dot2(1 << i, 100);
        cout << tester(Dot2(), 1 << i, 100).min_time << endl;
    }
    system("PAUSE");
    return 0;
}
