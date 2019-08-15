#pragma once
#define EIGEN_DONT_PARALLELIZE
//#define EIGEN_DONT_VECTORIZE
#include "utility.h"
#include <cassert>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <iterator>
#include <string>
#include <vector>

namespace eigen {
    using VectorX = Eigen::Matrix<scl_t, Eigen::Dynamic, 1>;
    using VecView = Eigen::Map<VectorX, Eigen::Aligned16>;
    template<int size>
    using MatView = Eigen::Map<Eigen::Matrix<scl_t, size, size, Eigen::RowMajor>, Eigen::Aligned16>;
        // >>>>>>>>> 1D tasks <<<<<<<<<<<

        struct BaselineSum {
        std::string id = "a = sum_i A_i";
        VecView A;
        scl_t ans;
        BaselineSum(int size, std::vector<scl_t*> const& data) :
            A(data[0], size), ans(eval()) {}
        scl_t eval() { return A.sum(); }
        bool check(float x) { return x == ans; }
    };

    struct BaselineSum_NoView {
        std::string id = "a = sum_i A_i";
        VectorX A;
        scl_t ans;
        BaselineSum_NoView(int size) : A(VectorX::Random(size)), ans(eval()) {}
        scl_t eval() { return A.sum(); }
        bool check(float x) { return x == ans; }
    };

    struct BaselineProd {
        std::string id = "B_i = c * A_i";
        VecView A;
        VectorX ans;
        scl_t c;
        BaselineProd(int size, std::vector<scl_t*> const& data) :
            A(data[0], size), c(random_scalar()), ans(eval()) {}
        VectorX eval() { return c * A; }
        bool check(VectorX x) { return x == ans; }
    };

    struct Dot {
        std::string id = "sum_i A_i * B_i";
        std::vector<VecView> A;
        scl_t ans;
        auto eval() { return A[0].dot(A[1]); }
        Dot(int size, std::vector<scl_t*> const& data) {
            std::for_each(data.begin(), data.begin() + 2, [&](scl_t* p) { A.emplace_back(p, size); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot1 {
        std::string id = "sum_i A_i * B_i * C_i";
        std::vector<VecView> A;
        scl_t ans;
        auto eval() { return A[0].cwiseProduct(A[1]).cwiseProduct(A[2]).sum(); }
        Dot1(int size, std::vector<scl_t*> const& data) {
            std::for_each(data.begin(), data.begin() + 3, [&](scl_t* p) { A.emplace_back(p, size); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot2 {
        std::string id = "sum_i ( A_i + B_i ) * C_i";
        std::vector<VecView> A;
        scl_t ans;
        auto eval() { return (A[0] + A[1]).dot(A[2]); }
        Dot2(int size, std::vector<scl_t*> const& data) {
            std::for_each(data.begin(), data.begin() + 3, [&](scl_t* p) { A.emplace_back(p, size); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot3 {
        std::string id = "sum_i (A_i + B_i) * (C_i - D_i)";
        std::vector<VecView> A;
        scl_t ans;
        auto eval() { return (A[0] + A[1]).dot(A[2] - A[3]); }
        Dot3(int size, std::vector<scl_t*> const& data) {
            std::for_each(data.begin(), data.begin() + 4, [&](scl_t* p) { A.emplace_back(p, size); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot4 {
        std::string id = "sum_i ( a * A_i + b * B_i ) * (c * C_i + d * D_i)";
        std::vector<VecView> A;
        std::array<scl_t, 4> a;
        scl_t ans;
        auto eval() { return (a[0] * A[0] + a[1] * A[1]).dot(a[2] * A[2] + a[3] * A[3]); }
        Dot4(int size, std::vector<scl_t*> const& data) {
            std::for_each(data.begin(), data.begin() + 4, [&](scl_t* p) { A.emplace_back(p, size); });
            std::generate(a.begin(), a.end(), [&]() {return random_scalar(); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot5 {
        std::string id = "sum_i ( A_i + B_i ) * (A_i  - C_i)";
        std::vector<VecView> A;
        scl_t ans;
        auto eval() { return (A[0] + A[1]).dot(A[0] - A[2]); }
        Dot5(int size, std::vector<scl_t*> const& data) {
            std::for_each(data.begin(), data.begin() + 3, [&](scl_t* p) { A.emplace_back(p, size); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot6 {
        std::string id = "sum_i ((A_i + B_i)) * ((C_i - D_i))";
        std::vector<VecView> A;
        scl_t ans;
        auto eval() {
            VectorX t1 = A[0] + A[1], t2 = A[2] - A[3];
            return t1.dot(t2);
        }
        Dot6(int size, std::vector<scl_t*> const& data) {
            std::for_each(data.begin(), data.begin() + 4, [&](scl_t* p) { A.emplace_back(p, size); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    // >>>>>>>>> 2D tasks <<<<<<<<<<<

    struct D2_1 {
        std::string id = "sum_i A_ij * B_i";
        std::array<scl_t, 0> scl;
        std::vector < vec, mat;
        scl_t ans;
        D2_1(int size, std::vector<scl_t> const& scls, std::vector<scl_t*> const& vecs, std::vector<scl_t*> const& mats) : size
            A(data[0], size), ans(eval()) {}
        scl_t eval() { return A.sum(); }
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