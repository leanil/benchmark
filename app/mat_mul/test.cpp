#include <algorithm>
#include <cstdlib>
#include <functional>
#include <random>
#include <utility>
#include <vector>
#include "mat_mul.hpp"

#include <iostream>

using namespace std;

struct Extents {
    int m, n, o;
};
vector<Extents> exts{{8, 8, 1},    {8, 8, 2},    {16, 16, 4},     {16, 16, 100},
                     {32, 32, 32}, {80, 80, 80}, {200, 200, 200}, {1000, 1000, 1000}};
using mat_mul = void (*)(double *, double *, double *, int, int, int);
vector<mat_mul> candidates{mat_mul_1,       mat_mul_2<2, 2>, mat_mul_2<2, 4>, mat_mul_2<4, 2>,
                           mat_mul_2<4, 4>, mat_mul_2<2, 8>, eigen_map, mat_mul_3<36,36,36,4,4>};
mat_mul baseline = mat_mul_0;

double abs_rel_diff(double x, double y) {
    double abs_err = abs(x - y), rel_err = abs_err / x;
    return min(abs_err, rel_err);
}

bool test(int id) {
    static default_random_engine engine;
    static uniform_real_distribution<> dist;
    static auto gen = bind(dist, ref(engine));
    static auto alloc = [&](int size) {
        return (double*)aligned_alloc(32, (size * sizeof(double) + 31) / 32 * 32);
    };
    mat_mul candidate = candidates[id];
    for (const Extents &ext : exts) {
        double *A = alloc(ext.m * ext.o), *B = alloc(ext.o * ext.n);
        double *C0 = alloc(ext.m * ext.n), *C1 = alloc(ext.m * ext.n);
        generate_n(A, ext.m * ext.o, gen);
        generate_n(B, ext.o * ext.n, gen);
        // for(int i=0;i<ext.m*ext.o;++i)
        //     A[i] = i;
        // for (int i = 0; i < ext.o * ext.n; ++i)
        //     B[i] = i;
        baseline(A, B, C0, ext.m, ext.n, ext.o);
        candidate(A, B, C1, ext.m, ext.n, ext.o);
        bool ok = true;
        for (int i = 0; i < ext.m * ext.n && ok; ++i)
            if (abs_rel_diff(C0[i], C1[i]) > 1e-6) ok = false;
        free(A), free(B), free(C0), free(C1);
        if (!ok) return false;
    }
    return true;
}

int main(int argc, char **argv) { 
    if(argc == 1) {
        exts.erase(exts.begin(), prev(exts.end()));
        return !test(6);
    }
    return !test(atoi(argv[1])); }