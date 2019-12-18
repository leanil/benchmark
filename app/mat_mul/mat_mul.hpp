#define EIGEN_DONT_PARALLELIZE
#include <x86intrin.h>
#include <Eigen/Dense>
#include <algorithm>

void mat_mul_0(double *A, double *B, double *C, int m, int n, int o) {  // A :: m x o, B :: o x n
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            C[i * n + j] = 0;
            for (int k = 0; k < o; ++k)
                C[i * n + j] += A[i * o + k] * B[k * n + j];
        }
}

void mat_mul_1(double *A, double *B, double *C, int m, int n, int o) {
    int bm = 2, bn = 2;
    for (int i = 0; i < m; i += bm)
        for (int j = 0; j < n; j += bn * 2) {
            __m128d c00, c01, c10, c11;
            c00 = c01 = c10 = c11 = _mm_setzero_pd();
            for (int k = 0; k < o; ++k) {
                __m128d a0 = _mm_load1_pd(A + i * o + k);
                __m128d a1 = _mm_load1_pd(A + (i + 1) * o + k);
                __m128d b0 = _mm_load_pd(B + k * n + j);
                __m128d b1 = _mm_load_pd(B + k * n + j + 2);
                c00 = _mm_fmadd_pd(a0, b0, c00);
                c01 = _mm_fmadd_pd(a0, b1, c01);
                c10 = _mm_fmadd_pd(a1, b0, c10);
                c11 = _mm_fmadd_pd(a1, b1, c11);
            }
            _mm_store_pd(C + i * n + j, c00);
            _mm_store_pd(C + i * n + j + 2, c01);
            _mm_store_pd(C + (i + 1) * n + j, c10);
            _mm_store_pd(C + (i + 1) * n + j + 2, c11);
        }
}

template <int bm, int bn>
void mat_mul_2(double *A, double *B, double *C, int m, int n, int o) {
    for (int i = 0; i < m; i += bm)
        for (int j = 0; j < n; j += bn) {
            __m128d c[bm][bn / 2];
            for (int i = 0; i < bm; ++i)
                for (int j = 0; j < bn / 2; ++j)
                    c[i][j] = _mm_setzero_pd();
            for (int k = 0; k < o; ++k) {
                __m128d a[bm], b[bn / 2];
                for (int l = 0; l < bm; ++l)
                    a[l] = _mm_load1_pd(A + (i + l) * o + k);
                for (int l = 0; l < bn / 2; ++l)
                    b[l] = _mm_load_pd(B + k * n + j + 2 * l);
                for (int i = 0; i < bm; ++i)
                    for (int j = 0; j < bn / 2; ++j)
                        c[i][j] = _mm_fmadd_pd(a[i], b[j], c[i][j]);
            }
            for (int k = 0; k < bm; ++k)
                for (int l = 0; l < bn / 2; ++l)
                    _mm_store_pd(C + (i + k) * n + j + 2 * l, c[k][l]);
        }
}

void eigen_map(double *a, double *b, double *c, int m, int n, int o) {
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Aligned32> A(a, m, o),
        B(b, o, n), C(c, m, n);
    C.noalias() = A * B;
}

template <int l1m, int l1n, int l1o, int rm, int rn>
void mat_mul_3(double *A, double *B, double *C, int m, int n, int o) {
    std::fill_n(C, n * m, 0);
    for (int i0 = 0; i0 < m; i0 += l1m)
        for (int j0 = 0; j0 < n; j0 += l1n)
            for (int k0 = 0; k0 < o; k0 += l1o)
                for (int i1 = 0; i1 < l1m && i1 < m - i0; i1 += rm)
                    for (int j1 = 0; j1 < l1n && j1 < n - j0; j1 += rn) {
                        __m128d c[rm][rn / 2];
                        int _rm = std::min(rm, m - i0 - i1), _rn = std::min(rn, n - j0 - j1);
                        for (int i2 = 0; i2 < _rm; ++i2)
                            for (int j2 = 0; j2 < _rn / 2; ++j2)
                                c[i2][j2] = _mm_load_pd(C + (i0 + i1 + i2) * n + (j0 + j1 + j2 * 2));
                        for (int k1 = 0; k1 < l1o && k1 < o - k0; ++k1) {
                            __m128d a[rm], b[rn / 2];
                            for (int i = 0; i < _rm; ++i)
                                a[i] = _mm_load1_pd(A + (i0 + i1 + i) * o + k0 + k1);
                            for (int i = 0; i < _rn / 2; ++i)
                                b[i] = _mm_load_pd(B + (k0 + k1) * n + j0 + j1 + i * 2);
                            for (int i2 = 0; i2 < _rm; ++i2)
                                for (int j2 = 0; j2 < _rn / 2; ++j2)
                                    c[i2][j2] = _mm_fmadd_pd(a[i2], b[j2], c[i2][j2]);
                        }
                        for (int i2 = 0; i2 < _rm; ++i2)
                            for (int j2 = 0; j2 < _rn / 2; ++j2)
                                _mm_store_pd(C + (i0 + i1 + i2) * n + j0 + j1 + j2 * 2, c[i2][j2]);
                    }
}

// swap at indices 0 1:
// permutation: 4
// rAB mA mB rAB
// 4 1 2 3
// template <int b, typename T0, typename T1, typename T2>
// auto kernel_lvl4(Int<3>, T0 &&res_, T1 &&A3_, T2 &&B3_) {
//     auto A3 = flip<0>(subdiv<1, b>(A3_));
//     auto B3 = flip<0>(subdiv<1, b>(B3_));
//     auto t00 = AllocateViewSimilarToButDropFrontDimensionsUpTo<0>(res_);
//     auto t12 = AllocateViewSimilarToButDropFrontDimensionsUpTo<2>(res_);
//     auto res = res_;
//     auto time0 = std::chrono::high_resolution_clock::now();
//     rnz(res, t12, lift(lift(add)),
//         [&](auto res, auto A2, auto B2) {
//             map(res,
//                 [&](auto res, auto A1) { map(res, [&](auto res, auto B1) { rnz(res, t00, add, mul, A1, B1); }, B2);
//                 }, A2);
//         },
//         A3, B3);
//     auto time1 = std::chrono::high_resolution_clock::now();
//     return ms(time0, time1);
// }