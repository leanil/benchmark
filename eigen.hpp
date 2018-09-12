#pragma once
#define EIGEN_DONT_PARALLELIZE
//#define EIGEN_DONT_VECTORIZE
#include "utility.h"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <string>

using VectorX = Eigen::Matrix<scl_t, Eigen::Dynamic, 1>;
using View = Eigen::Map<Eigen::Matrix<scl_t, Eigen::Dynamic, 1>, Eigen::Aligned32>;

struct BaselineSum {
    std::string id = "a = sum_i A_i";
    VectorX A;
    scl_t ans;
    void init(int size) {
        A = VectorX::Random(size);
        ans = eval();
    }
    scl_t eval() { return A.sum(); }
    bool check(float x) { return x == ans; }
};

struct BaselineProd {
    std::string id = "B_i = c * A_i";
    VectorX A, ans;
    scl_t c;
    void init(int size) {
        A = VectorX::Random(size);
        c = random_scalar();
        ans = eval();
    }
    VectorX eval() { return c * A; }
    bool check(VectorX x) { return x == ans; }
};

struct Dot {
    std::string id = "sum_i A_i * B_i";
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

struct Dot1 {
    std::string id = "sum_i A_i * B_i * C_i";
    std::array<VectorX, 3> A;
    scl_t ans;
    auto eval() { return A[0].cwiseProduct(A[1]).cwiseProduct(A[2]).sum(); }
    void init(int size) {
        std::generate(A.begin(), A.end(), [&]() {return VectorX::Random(size); });
        ans = eval();
    }
    bool check(float x) { return x == ans; }
};

struct Dot2 {
    std::string id = "sum_i ( A_i + B_i ) * C_i";
    std::array<VectorX, 3> A;
    scl_t ans;
    auto eval() { return (A[0] + A[1]).dot(A[2]); }
    void init(int size) {
        std::generate(A.begin(), A.end(), [&]() {return VectorX::Random(size); });
        ans = eval();
    }
    bool check(float x) { return x == ans; }
};

struct Dot3 {
    std::string id = "sum_i (A_i + B_i) * (C_i - D_i)";
    std::array<VectorX, 4> A;
    scl_t ans;
    auto eval() { return (A[0] + A[1]).dot(A[2] - A[3]); }
    void init(int size) {
        std::generate(A.begin(), A.end(), [&]() {return VectorX::Random(size); });
        ans = eval();
    }
    bool check(float x) { return x == ans; }
};

struct Dot4 {
    std::string id = "sum_i ( a * A_i + b * B_i ) * (c * C_i + d * D_i)";
    std::array<VectorX, 4> A;
    std::array<scl_t, 4> a;
    scl_t ans;
    auto eval() { return (a[0] * A[0] + a[1] * A[1]).dot(a[2] * A[2] + a[3] * A[3]); }
    void init(int size) {
        std::generate(A.begin(), A.end(), [&]() {return VectorX::Random(size); });
        std::generate(a.begin(), a.end(), [&]() {return random_scalar(); });
        ans = eval();
    }
    bool check(float x) { return x == ans; }
};

struct Dot5 {
    std::string id = "sum_i ( A_i + B_i ) * (A_i  - C_i)";
    std::array<VectorX, 3> A;
    scl_t ans;
    auto eval() { return (A[0] + A[1]).dot(A[0] - A[2]); }
    void init(int size) {
        std::generate(A.begin(), A.end(), [&]() {return VectorX::Random(size); });
        ans = eval();
    }
    bool check(float x) { return x == ans; }
};

struct Dot6 {
    std::string id = "sum_i ((A_i + B_i)) * ((C_i - D_i))";
    std::array<VectorX, 4> A;
    scl_t ans;
    auto eval() { 
        VectorX t1 = A[0] + A[1], t2 = A[2] - A[3];
        return t1.dot(t2);
    }
    void init(int size) {
        std::generate(A.begin(), A.end(), [&]() {return VectorX::Random(size); });
        ans = eval();
    }
    bool check(float x) { return x == ans; }
};