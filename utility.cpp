#include "utility.h"
#include <algorithm>
#include <chrono>
#include <random>

using namespace std;

auto random_vector(int size) {
    vector<scl_t> vec(size);
    static default_random_engine gen;
    uniform_real_distribution<scl_t> dist;
    generate(vec.begin(), vec.end(), [&]() {return dist(gen); });
    return vec;
}

scl_t random_scalar() {
    static default_random_engine gen;
    uniform_real_distribution<scl_t> dist;
    return dist(gen);
}
