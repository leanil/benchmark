#pragma once
#define EIGEN_DONT_PARALLELIZE
#include "utility.h"
#include "Eigen/Dense"
#include <algorithm>
#include <array>

using VectorX = Eigen::Matrix<scl_t, Eigen::Dynamic, 1>;

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
struct Dot2 {
    std::array<VectorX, 3> A;
    scl_t ans;
    auto eval() { return (A[0] + A[1]).dot(A[2]); }
    void init(int size) {
        generate(A.begin(), A.end(), [&]() {return VectorX::Random(size); });
        ans = eval();
    }
    bool check(float x) { return x == ans; }
};

// sum_i (A_i + B_i) * (C_i - D_i)
struct Dot3 {
    std::array<VectorX, 4> A;
    scl_t ans;
    auto eval() { return (A[0] + A[1]).dot(A[2] - A[3]); }
    void init(int size) {
        generate(A.begin(), A.end(), [&]() {return VectorX::Random(size); });
        ans = eval();
    }
    bool check(float x) { return x == ans; }
};

// sum_i ( a * A_i + b * B_i ) * (c * C_i + d * D_i)
struct Dot4 {
    std::array<VectorX, 4> A;
    std::array<scl_t, 4> a;
    scl_t ans;
    auto eval() { return (a[0] * A[0] + a[1] * A[1]).dot(a[2] * A[2] + a[3] * A[3]); }
    void init(int size) {
        generate(A.begin(), A.end(), [&]() {return VectorX::Random(size); });
        generate(a.begin(), a.end(), [&]() {return random_scalar(); });
        ans = eval();
    }
    bool check(float x) { return x == ans; }
};