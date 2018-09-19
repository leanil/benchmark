#pragma once
#include "utility.h"
#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <vector>

namespace naive {

    struct BaselineSum {
        std::string id = "a = sum_i A_i";
        std::array<scl_t*, 1> A;
        int size;
        scl_t ans;
        BaselineSum(int size, std::vector<scl_t*> const& data) : A{ data[0] }, size(size), ans(eval()) {}
        scl_t eval() {
            scl_t sum = 0;
            for (int i = 0; i < size; ++i)
                sum += A[0][i];
            return sum;
        }
        bool check(float x) { return x == ans; }
    };

    struct BaselineProd {
        std::string id = "B_i = c * A_i";
        std::array<scl_t*, 1> A;
        int size;
        scl_t c;
        std::vector<scl_t> ans;
        BaselineProd(int size, std::vector<scl_t*> const& data) :
            A{ data[0] }, size(size), c(random_scalar()), ans(eval()) {}
        std::vector<scl_t> eval() {
            std::vector<scl_t> B(size);
            for (int i = 0; i < size; ++i)
                B[i] = c * A[0][i];
            return B;
        }
        bool check(std::vector<scl_t> x) { return x == ans; }
    };

    struct Dot {
        std::string id = "sum_i A_i * B_i";
        std::array<scl_t*, 2> A;
        int size;
        scl_t ans;
        auto eval() {
            scl_t sum = 0;
            for (int i = 0; i < size; ++i)
                sum += A[0][i] * A[1][i];
            return sum;
        }
        Dot(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot1 {
        std::string id = "sum_i A_i * B_i * C_i";
        int size;
        std::array<scl_t*, 3> A;
        scl_t ans;
        auto eval() {
            scl_t sum = 0;
            for (int i = 0; i < size; ++i)
                sum += A[0][i] * A[1][i] * A[2][i];
            return sum;
        }
        Dot1(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot2 {
        std::string id = "sum_i ( A_i + B_i ) * C_i";
        std::array<scl_t*, 3> A;
        int size;
        scl_t ans;
        auto eval() {
            scl_t sum = 0;
            for (int i = 0; i < size; ++i)
                sum += (A[0][i] + A[1][i]) * A[2][i];
            return sum;
        }
        Dot2(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot3 {
        std::string id = "sum_i (A_i + B_i) * (C_i - D_i)";
        std::array<scl_t*, 4> A;
        int size;
        scl_t ans;
        auto eval() {
            scl_t sum = 0;
            for (int i = 0; i < size; ++i)
                sum += (A[0][i] + A[1][i]) * (A[2][i] - A[3][i]);
            return sum;
        }
        Dot3(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot4 {
        std::string id = "sum_i ( a * A_i + b * B_i ) * (c * C_i + d * D_i)";
        std::array<scl_t*, 4> A;
        int size;
        std::array<scl_t, 4> a;
        scl_t ans;
        auto eval() {
            scl_t sum = 0;
            for (int i = 0; i < size; ++i)
                sum += (a[0] * A[0][i] + a[1] * A[1][i]) * (a[2] * A[2][i] + a[3] * A[3][i]);
            return sum;
        }
        Dot4(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            std::generate(a.begin(), a.end(), [&]() {return random_scalar(); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot5 {
        std::string id = "sum_i ( A_i + B_i ) * (A_i  - C_i)";
        std::array<scl_t*, 3> A;
        int size;
        scl_t ans;
        auto eval() {
            scl_t sum = 0;
            for (int i = 0; i < size; ++i)
                sum += (A[0][i] + A[1][i]) * (A[0][i] - A[2][i]);
            return sum;
        }
        Dot5(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot6 {
        std::string id = "sum_i ((A_i + B_i)) * ((C_i - D_i))";
        std::array<scl_t*, 4> A;
        int size;
        scl_t ans;
        auto eval() {
            std::vector<scl_t> t1(size), t2(size);
            for (int i = 0; i < size; ++i)
                t1[i] = A[0][i] + A[1][i];
            for (int i = 0; i < size; ++i)
                t2[i] = A[2][i] - A[3][i];
            scl_t sum = 0;
            for (int i = 0; i < size; ++i)
                sum += t1[i] * t2[i];
            return sum;
        }
        Dot6(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    std::array<std::function<TestResult(int, std::vector<scl_t*> const&, int)>, 8> testers{
        test_helper<BaselineSum>,
        test_helper<Dot>,
        test_helper<Dot1>,
        test_helper<Dot2>,
        test_helper<Dot3>,
        test_helper<Dot4>,
        test_helper<Dot5>,
        test_helper<Dot6>
    };

}