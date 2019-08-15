#pragma once
#include "utility.h"
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace vectorized {

    float sum(__m128& ps) {
        ps = _mm_hadd_ps(ps, ps);
        ps = _mm_hadd_ps(ps, ps);
        //ps = _mm_hadd_ps(ps, ps);
        return _mm_cvtss_f32(ps);
    }

    struct BaselineSum {
        std::string id = "a = sum_i A_i";
        std::array<scl_t*, 1> A;
        int size;
        scl_t ans;
        BaselineSum(int size, std::vector<scl_t*> const& data) : A{ data[0] }, size(size), ans(eval()) {}
        scl_t eval() {
            __m128 mmSum = _mm_setzero_ps();
            int i = 0;
            for (; i + 3 < size; i += 4)
                mmSum = _mm_add_ps(mmSum, _mm_load_ps(A[0] + i));
            float ret = sum(mmSum);
            for (; i < size; ++i)
                ret += A[0][i];
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    //struct BaselineProd {
    //    std::string id = "B_i = c * A_i";
    //    std::vector<scl_t> A, ans;
    //    scl_t c;
    //    void init(int size) {
    //        A = random_vector(size);
    //        c = random_scalar();
    //        ans = eval();
    //    }
    //    std::vector<scl_t> eval() {
    //        std::vector<scl_t> B(A.size());
    //        __m128 C = _mm_broadcast_ss(&c);
    //        const int n = (int)A.size();
    //        const scl_t* p = A.data();
    //        scl_t* q = B.data();
    //        int i = 0;
    //        for (; i + 7 < n; i += 8)
    //            _mm_storeu_ps(q + i, _mm_mul_ps(C, _mm_loadu_ps(p + i)));
    //        int mask_data[8]{};
    //        for (int j = n % 8; j < 8; ++j)
    //            mask_data[j] = -1;
    //        __m128i mask = _mm_loadu_si128((__m128i*)mask_data);
    //        _mm_maskstore_ps(q + i, mask, _mm_mul_ps(C, _mm_maskload_ps(p + i, mask)));
    //        return B;
    //    }
    //    bool check(std::vector<scl_t> x) { return x == ans; }
    //};

    struct Dot {
        std::string id = "sum_i A_i * B_i";
        std::array<scl_t*, 2> A;
        int size;
        scl_t ans;
        Dot(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            ans = eval();
        }
        scl_t eval() {
            __m128 mmSum = _mm_setzero_ps();
            int i = 0;
            for (; i + 3 < size; i += 4)
                mmSum = _mm_add_ps(mmSum, _mm_mul_ps(_mm_load_ps(A[0] + i), _mm_load_ps(A[1] + i)));
            float ret = sum(mmSum);
            for (; i < size; ++i)
                ret += A[0][i] * A[1][i];
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot1 {
        std::string id = "sum_i A_i * B_i * C_i";
        std::array<scl_t*, 3> A;
        int size;
        scl_t ans;
        Dot1(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            ans = eval();
        }
        scl_t eval() {
            __m128 mmSum = _mm_setzero_ps();
            int i = 0;
            for (; i + 3 < size; i += 4)
                mmSum = _mm_add_ps(mmSum, _mm_mul_ps(
                    _mm_mul_ps(_mm_load_ps(A[0] + i), _mm_load_ps(A[1] + i)),
                    _mm_load_ps(A[2] + i)));
            float ret = sum(mmSum);
            for (; i < size; ++i)
                ret += A[0][i] * A[1][i] * A[2][i];
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot2 {
        std::string id = "sum_i ( A_i + B_i ) * C_i";
        std::array<scl_t*, 3> A;
        int size;
        scl_t ans;
        Dot2(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            ans = eval();
        }
        scl_t eval() {
            __m128 mmSum = _mm_setzero_ps();
            int i = 0;
            for (; i + 3 < size; i += 4)
                mmSum = _mm_add_ps(mmSum, _mm_mul_ps(
                    _mm_add_ps(_mm_load_ps(A[0] + i), _mm_load_ps(A[1] + i)),
                    _mm_load_ps(A[2] + i)));
            float ret = sum(mmSum);
            for (; i < size; ++i)
                ret += (A[0][i] + A[1][i]) * A[2][i];
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot3 {
        std::string id = "sum_i (A_i + B_i) * (C_i - D_i)";
        std::array<scl_t*, 4> A;
        int size;
        scl_t ans;
        Dot3(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            ans = eval();
        }
        scl_t eval() {
            __m128 mmSum = _mm_setzero_ps();
            int i = 0;
            for (; i + 3 < size; i += 4)
                mmSum = _mm_add_ps(mmSum, _mm_mul_ps(
                    _mm_add_ps(_mm_load_ps(A[0] + i), _mm_load_ps(A[1] + i)),
                    _mm_sub_ps(_mm_load_ps(A[2] + i), _mm_load_ps(A[3] + i))));
            float ret = sum(mmSum);
            for (; i < size; ++i)
                ret += (A[0][i] + A[1][i]) * (A[2][i] - A[3][i]);
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot4 {
        std::string id = "sum_i ( a * A_i + b * B_i ) * (c * C_i + d * D_i)";
        std::array<scl_t*, 4> A;
        std::array<scl_t, 4> x;
        int size;
        scl_t ans;
        Dot4(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            std::generate(x.begin(), x.end(), [&]() {return random_scalar(); });
            ans = eval();
        }
        scl_t eval() {
            std::array<__m128, 4> a;
            std::transform(x.begin(), x.end(), a.begin(), [](scl_t s) { return _mm_set1_ps(s); });
            __m128 mmSum = _mm_setzero_ps();
            int i = 0;
            for (; i + 3 < size; i += 4)
                mmSum = _mm_add_ps(mmSum, _mm_mul_ps(
                    _mm_add_ps(_mm_mul_ps(a[0], _mm_load_ps(A[0] + i)), _mm_mul_ps(a[1], _mm_load_ps(A[1] + i))),
                    _mm_add_ps(_mm_mul_ps(a[2], _mm_load_ps(A[2] + i)), _mm_mul_ps(a[3], _mm_load_ps(A[3] + i)))));
            float ret = sum(mmSum);
            for (; i < size; ++i)
                ret += (x[0] * A[0][i] + x[1] * A[1][i]) * (x[2] * A[2][i] - x[3] * A[3][i]);
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot5 {
        std::string id = "sum_i ( A_i + B_i ) * (A_i  - C_i)";
        std::array<scl_t*, 3> A;
        int size;
        scl_t ans;
        Dot5(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            ans = eval();
        }
        scl_t eval() {
            __m128 mmSum = _mm_setzero_ps();
            int i = 0;
            for (; i + 3 < size; i += 4)
                mmSum = _mm_add_ps(mmSum, _mm_mul_ps(
                    _mm_add_ps(_mm_load_ps(A[0] + i), _mm_load_ps(A[1] + i)),
                    _mm_sub_ps(_mm_load_ps(A[0] + i), _mm_load_ps(A[2] + i))));
            float ret = sum(mmSum);
            for (; i < size; ++i)
                ret += (A[0][i] + A[1][i]) * (A[0][i] - A[2][i]);
            return ret;
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot6 {
        std::string id = "sum_i ((A_i + B_i)) * ((C_i - D_i))";
        std::array<scl_t*, 4> A;
        int size;
        scl_t ans;
        Dot6(int size, std::vector<scl_t*> const& data) : size(size) {
            std::copy_n(data.begin(), A.size(), A.begin());
            ans = eval();
        }
        scl_t eval() {
            std::vector<scl_t> T1(size), T2(size);
            scl_t *t1 = T1.data(), *t2 = T2.data();
            int i = 0;
            for (; i + 3 < size; i += 4) {
                _mm_store_ps(t1 + i, _mm_add_ps(_mm_load_ps(A[0] + i), _mm_load_ps(A[1] + i)));
                _mm_store_ps(t2 + i, _mm_sub_ps(_mm_load_ps(A[2] + i), _mm_load_ps(A[3] + i)));
            }
            for (; i < size; ++i) {
                t1[i] = A[0][i] + A[1][i];
                t2[i] = A[2][i] - A[3][i];
            }
            __m128 mmSum = _mm_setzero_ps();
            i = 0;
            for (; i + 3 < size; i += 4)
                mmSum = _mm_add_ps(mmSum, _mm_mul_ps(_mm_load_ps(t1 + i), _mm_load_ps(t2 + i)));
            float ret = sum(mmSum);
            for (; i < size; ++i)
                ret += t1[i] * t2[i];
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