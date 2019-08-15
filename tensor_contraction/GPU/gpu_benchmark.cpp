#include "eigen/eigen_test.h"
#include "futhark/futhark_test.hpp"
#include <cstdlib>
#include <random>
#include <string>
#include <iomanip>

using namespace std;

bool compare(scl_t* a, scl_t* b, int n, string const& test) {
    static const scl_t eps = 1e-5;
    for (int i = 0; i < n; ++i)
        if (abs(a[i] - b[i]) > eps) {
            cout << "output mismatch in test " << test << endl;
            return false;
        }
    cout << "test " << test << " OK" << endl;
    return true;
}

template<int p>
bool functional_test_1d() {
    scl_t *data[4], scl[4], out[2];
    default_random_engine gen(42);
    uniform_real_distribution<scl_t> dist;
    for (int i = 0; i < 4; ++i) {
        data[i] = allocate<p>();
        scl[i] = dist(gen);
    }
    bool success = true;
    
    eigen::baseline_sum<p>(data[0], &out[0]);
    futhark::baseline_sum<p>(data[0], &out[1]);
    success &= compare(&out[0], &out[1], 1, "baseline_sum");

    eigen::dot<p>(data[0], data[1], &out[0]);
    futhark::dot<p>(data[0], data[1], &out[1]);
    success &= compare(&out[0], &out[1], 1, "dot");

    eigen::dot1<p>(data[0], data[1], data[2], &out[0]);
    futhark::dot1<p>(data[0], data[1], data[2], &out[1]);
    success &= compare(&out[0], &out[1], 1, "dot1");

    eigen::dot2<p>(data[0], data[1], data[2], &out[0]);
    futhark::dot2<p>(data[0], data[1], data[2], &out[1]);
    success &= compare(&out[0], &out[1], 1, "dot2");

    eigen::dot3<p>(data[0], data[1], data[2], &out[0]);
    futhark::dot3<p>(data[0], data[1], data[2], &out[1]);
    success &= compare(&out[0], &out[1], 1, "dot3");

    eigen::dot4<p>(data[0], data[1], data[2], data[3], &out[0]);
    futhark::dot4<p>(data[0], data[1], data[2], data[3], &out[1]);
    success &= compare(&out[0], &out[1], 1, "dot4");

    eigen::dot5<p>(&scl[0], data[0], &scl[1], data[1], &scl[2], data[2], &scl[3], data[3], &out[0]);
    futhark::dot5<p>(&scl[0], data[0], &scl[1], data[1], &scl[2], data[2], &scl[3], data[3], &out[1]);
    success &= compare(&out[0], &out[1], 1, "dot5");

    eigen::dot6<p>(data[0], data[1], data[2], data[3], &out[0]);
    futhark::dot6<p>(data[0], data[1], data[2], data[3], &out[1]);
    success &= compare(&out[0], &out[1], 1, "dot6");

    for (int i = 0; i < 4; ++i)
        delete[] data[i];
    return success;
}

int main() {
    cout << (functional_test_1d<10>() ? "functional test successful" : "functional test failed") << endl;

    /*cout << "eigen warmup: " << eigen::baseline_sum<10000000>() << "\n\n";
    cout << "futhark warmup: " << futhark::baseline_sum<10000000>() << "\n\n";

    cout << "Baseline:\n";
    cout << "sum\n";
    for (int i = 0; i < 10; ++i)
        cout << eigen::baseline_sum<10000000>() << ' ' << futhark::baseline_sum<10000000>() << endl;
    cout << "\nprod\n";
    for (int i = 0; i < 10; ++i)
        cout << eigen::baseline_prod<10000000>() << ' ' << futhark::baseline_prod<10000000>() << endl;*/

}