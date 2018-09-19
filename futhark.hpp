#pragma once
#include "utility.h"
#include "futhark_lib.h"
#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <vector>

namespace futhark {

    futhark_context_config* config = futhark_context_config_new();

    struct BaselineSum {
        std::string id = "a = sum_i A_i";
        std::array<futhark_f32_1d*, 1> A;
        int size;
        scl_t ans;
        futhark_context* context;
        BaselineSum(int size, std::vector<scl_t*> const& data) : size(size) {
            context = futhark_context_new(config);
            std::transform(data.begin(), data.begin() + A.size(), A.begin(),
                [&](scl_t* p) {return futhark_new_f32_1d(context, p, size); });
            ans = eval();
        }
        ~BaselineSum() {
            std::for_each(A.begin(), A.end(), [&](futhark_f32_1d* p) {futhark_free_f32_1d(context, p); });
            futhark_context_free(context);
        }
        scl_t eval() {
            scl_t ret;
            futhark_entry_BaselineSum(context, &ret, A[0]);
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    //struct BaselineProd {
    //    std::string id = "B_i = c * A_i";
    //    std::array<scl_t*, 1> A;
    //    int size;
    //    scl_t c;
    //    std::vector<scl_t> ans;
    //    BaselineProd(int size, std::vector<scl_t*> const& data) :
    //        A{ data[0] }, size(size), c(random_scalar()), ans(eval()) {}
    //    std::vector<scl_t> eval() {
    //        std::vector<scl_t> B(size);
    //        for (int i = 0; i < size; ++i)
    //            B[i] = c * A[0][i];
    //        return B;
    //    }
    //    bool check(std::vector<scl_t> x) { return x == ans; }
    //};

    struct Dot {
        std::string id = "sum_i A_i * B_i";
        std::array<futhark_f32_1d*, 2> A;
        int size;
        scl_t ans;
        futhark_context* context;
        Dot(int size, std::vector<scl_t*> const& data) : size(size) {
            context = futhark_context_new(config);
            std::transform(data.begin(), data.begin() + A.size(), A.begin(),
                [&](scl_t* p) {return futhark_new_f32_1d(context, p, size); });
            ans = eval();
        }
        ~Dot() {
            std::for_each(A.begin(), A.end(), [&](futhark_f32_1d* p) {futhark_free_f32_1d(context, p); });
            futhark_context_free(context);
        }
        scl_t eval() {
            scl_t ret;
            futhark_entry_Dot(context, &ret, A[0], A[1]);
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot1 {
        std::string id = "sum_i A_i * B_i * C_i";
        std::array<futhark_f32_1d*, 3> A;
        int size;
        scl_t ans;
        futhark_context* context;
        Dot1(int size, std::vector<scl_t*> const& data) : size(size) {
            context = futhark_context_new(config);
            std::transform(data.begin(), data.begin() + A.size(), A.begin(),
                [&](scl_t* p) {return futhark_new_f32_1d(context, p, size); });
            ans = eval();
        }
        ~Dot1() {
            std::for_each(A.begin(), A.end(), [&](futhark_f32_1d* p) {futhark_free_f32_1d(context, p); });
            futhark_context_free(context);
        }
        scl_t eval() {
            scl_t ret;
            futhark_entry_Dot1(context, &ret, A[0], A[1], A[2]);
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot2 {
        std::string id = "sum_i ( A_i + B_i ) * C_i";
        std::array<futhark_f32_1d*, 3> A;
        int size;
        scl_t ans;
        futhark_context* context;
        Dot2(int size, std::vector<scl_t*> const& data) : size(size) {
            context = futhark_context_new(config);
            std::transform(data.begin(), data.begin() + A.size(), A.begin(),
                [&](scl_t* p) {return futhark_new_f32_1d(context, p, size); });
            ans = eval();
        }
        ~Dot2() {
            std::for_each(A.begin(), A.end(), [&](futhark_f32_1d* p) {futhark_free_f32_1d(context, p); });
            futhark_context_free(context);
        }
        scl_t eval() {
            scl_t ret;
            futhark_entry_Dot2(context, &ret, A[0], A[1], A[2]);
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot3 {
        std::string id = "sum_i (A_i + B_i) * (C_i - D_i)";
        std::array<futhark_f32_1d*, 4> A;
        int size;
        scl_t ans;
        futhark_context* context;
        Dot3(int size, std::vector<scl_t*> const& data) : size(size) {
            context = futhark_context_new(config);
            std::transform(data.begin(), data.begin() + A.size(), A.begin(),
                [&](scl_t* p) {return futhark_new_f32_1d(context, p, size); });
            ans = eval();
        }
        ~Dot3() {
            std::for_each(A.begin(), A.end(), [&](futhark_f32_1d* p) {futhark_free_f32_1d(context, p); });
            futhark_context_free(context);
        }
        scl_t eval() {
            scl_t ret;
            futhark_entry_Dot3(context, &ret, A[0], A[1], A[2], A[3]);
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot4 {
        std::string id = "sum_i ( a * A_i + b * B_i ) * (c * C_i + d * D_i)";
        std::array<futhark_f32_1d*, 4> A;
        std::array<scl_t, 4> a;
        int size;
        scl_t ans;
        futhark_context* context;
        Dot4(int size, std::vector<scl_t*> const& data) : size(size) {
            context = futhark_context_new(config);
            std::transform(data.begin(), data.begin() + A.size(), A.begin(),
                [&](scl_t* p) {return futhark_new_f32_1d(context, p, size); });
            std::generate(a.begin(), a.end(), [&]() {return random_scalar(); });
            ans = eval();
        }
        ~Dot4() {
            std::for_each(A.begin(), A.end(), [&](futhark_f32_1d* p) {futhark_free_f32_1d(context, p); });
            futhark_context_free(context);
        }
        scl_t eval() {
            scl_t ret;
            futhark_entry_Dot4(context, &ret, a[0], A[0], a[1], A[1], a[2], A[2], a[3], A[3]);
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot5 {
        std::string id = "sum_i ( A_i + B_i ) * (A_i  - C_i)";
        std::array<futhark_f32_1d*, 3> A;
        int size;
        scl_t ans;
        futhark_context* context;
        Dot5(int size, std::vector<scl_t*> const& data) : size(size) {
            context = futhark_context_new(config);
            std::transform(data.begin(), data.begin() + A.size(), A.begin(),
                [&](scl_t* p) {return futhark_new_f32_1d(context, p, size); });
            ans = eval();
        }
        ~Dot5() {
            std::for_each(A.begin(), A.end(), [&](futhark_f32_1d* p) {futhark_free_f32_1d(context, p); });
            futhark_context_free(context);
        }
        scl_t eval() {
            scl_t ret;
            futhark_entry_Dot5(context, &ret, A[0], A[1], A[2]);
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot6 {
        std::string id = "sum_i ((A_i + B_i)) * ((C_i - D_i))";
        std::array<futhark_f32_1d*, 4> A;
        int size;
        scl_t ans;
        futhark_context* context;
        Dot6(int size, std::vector<scl_t*> const& data) : size(size) {
            context = futhark_context_new(config);
            std::transform(data.begin(), data.begin() + A.size(), A.begin(),
                [&](scl_t* p) {return futhark_new_f32_1d(context, p, size); });
            ans = eval();
        }
        ~Dot6() {
            std::for_each(A.begin(), A.end(), [&](futhark_f32_1d* p) {futhark_free_f32_1d(context, p); });
            futhark_context_free(context);
        }
        scl_t eval() {
            scl_t ret;
            futhark_entry_Dot6(context, &ret, A[0], A[1], A[2], A[3]);
            return ret;
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