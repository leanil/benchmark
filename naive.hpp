#pragma once
#include "utility.h"
#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace naive {

    struct BaselineSum {
        std::string id = "a = sum_i A_i";
        std::vector<scl_t> A;
        scl_t ans;
        void init(int size) {
            A = random_vector(size);
            ans = eval();
        }
        scl_t eval() {
            scl_t sum = 0;
            for (int i = 0; i < (int)A.size(); ++i)
                sum += A[i];
            return sum;
        }
        bool check(float x) { return x == ans; }
    };

    struct BaselineProd {
        std::string id = "B_i = c * A_i";
        std::vector<scl_t> A, ans;
        scl_t c;
        void init(int size) {
            A = random_vector(size);
            c = random_scalar();
            ans = eval();
        }
        std::vector<scl_t> eval() {
            std::vector<scl_t> B(A.size());
                for (int i = 0; i < (int)A.size(); ++i)
                    B[i] = c * A[i];
            return B;
        }
        bool check(std::vector<scl_t> x) { return x == ans; }
    };

    struct Dot {
        std::string id = "sum_i A_i * B_i";
        std::vector<scl_t> A, B;
        scl_t ans;
        auto eval() {
            scl_t sum = 0;
            for (int i = 0; i < (int)A.size(); ++i)
                sum += A[i] * B[i];
            return sum;
        }
        void init(int size) {
            A = random_vector(size);
            B = random_vector(size);
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot1 {
        std::string id = "sum_i A_i * B_i * C_i";
        std::array<std::vector<scl_t>, 3> A;
        scl_t ans;
        auto eval() {
            scl_t sum = 0;
            for (int i = 0; i < (int)A[0].size(); ++i)
                sum += A[0][i] * A[1][i] * A[2][i];
            return sum;
        }
        void init(int size) {
            generate(A.begin(), A.end(), [&]() {return random_vector(size); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot2 {
        std::string id = "sum_i ( A_i + B_i ) * C_i";
        std::array<std::vector<scl_t>, 3> A;
        scl_t ans;
        auto eval() {
            scl_t sum = 0;
            for (int i = 0; i < (int)A[0].size(); ++i)
                sum += (A[0][i] + A[1][i]) * A[2][i];
            return sum;
        }
        void init(int size) {
            generate(A.begin(), A.end(), [&]() {return random_vector(size); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot3 {
        std::string id = "sum_i (A_i + B_i) * (C_i - D_i)";
        std::array<std::vector<scl_t>, 4> A;
        scl_t ans;
        auto eval() {
            scl_t sum = 0;
            for (int i = 0; i < (int)A[0].size(); ++i)
                sum += (A[0][i] + A[1][i]) * (A[2][i] - A[3][i]);
            return sum;
        }
        void init(int size) {
            generate(A.begin(), A.end(), [&]() {return random_vector(size); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot4 {
        std::string id = "sum_i ( a * A_i + b * B_i ) * (c * C_i + d * D_i)";
        std::array<std::vector<scl_t>, 4> A;
        std::array<scl_t, 4> a;
        scl_t ans;
        auto eval() {
            scl_t sum = 0;
            for (int i = 0; i < (int)A[0].size(); ++i)
                sum += (a[0] * A[0][i] + a[1] * A[1][i]) * (a[2] * A[2][i] + a[3] * A[3][i]);
            return sum;
        }
        void init(int size) {
            generate(A.begin(), A.end(), [&]() {return random_vector(size); });
            generate(a.begin(), a.end(), [&]() {return random_scalar(); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot5 {
        std::string id = "sum_i ( A_i + B_i ) * (A_i  - C_i)";
        std::array<std::vector<scl_t>, 3> A;
        scl_t ans;
        auto eval() {
            scl_t sum = 0;
            for (int i = 0; i < (int)A[0].size(); ++i)
                sum += (A[0][i] + A[1][i]) * (A[0][i] - A[2][i]);
            return sum;
        }
        void init(int size) {
            generate(A.begin(), A.end(), [&]() {return random_vector(size); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };

    struct Dot6 {
        std::string id = "sum_i ((A_i + B_i)) * ((C_i - D_i))";
        std::array<std::vector<scl_t>, 4> A;
        scl_t ans;
        auto eval() {
            std::vector<scl_t> t1(A[0].size()), t2(A[2].size());
            for (int i = 0; i < (int)A[0].size(); ++i)
                t1[i] = A[0][i] + A[1][i];
            for (int i = 0; i < (int)A[2].size(); ++i)
                t2[i] = A[2][i] + A[3][i];
            scl_t sum = 0;
            for (int i = 0; i < (int)A[0].size(); ++i)
                sum += t1[i] * t2[i];
            return sum;
        }
        void init(int size) {
            generate(A.begin(), A.end(), [&]() {return random_vector(size); });
            ans = eval();
        }
        bool check(float x) { return x == ans; }
    };
}