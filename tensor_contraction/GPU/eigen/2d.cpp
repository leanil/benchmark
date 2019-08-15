#include "2d.h"
#include "2d.hpp"

namespace eigen {

    long long t1(scl_t* A_host, scl_t* B_host, scl_t* ans_host) {
        return t1<i_2d, j_2d>(A_host, B_host, ans_host);
    }

    long long t2(scl_t* A_host, scl_t* B_host, scl_t* C_host, scl_t* ans_host) {
        return t2<i_2d, j_2d>(A_host, B_host, C_host, ans_host);
    }

    long long t3(scl_t* A_host, scl_t* B_host, scl_t* C_host, scl_t* ans_host) {
        return t3<i_2d, j_2d>(A_host, B_host, C_host, ans_host);
    }

    long long t4(scl_t* A_host, scl_t* B_host, scl_t* C_host, scl_t* D_host, scl_t* ans_host) {
        return t4<i_2d, j_2d>(A_host, B_host, C_host, D_host, ans_host);
    }

    long long t5(scl_t* a, scl_t* A_host, scl_t* b, scl_t* B_host, scl_t* c, scl_t* C_host, scl_t* d, scl_t* D_host, scl_t* ans_host) {
        return t5<i_2d, j_2d>(a, A_host, b, B_host, c, C_host, d, D_host, ans_host);
    }

    long long t6(scl_t* A_host, scl_t* B_host, scl_t* C_host, scl_t* D_host, scl_t* ans_host) {
        return t6<i_2d, j_2d>(A_host, B_host, C_host, D_host, ans_host);
    }

    long long t7(scl_t* A_host, scl_t* C_host, scl_t* D_host, scl_t* ans_host) {
        return t7<i_2d, j_2d, k_2d>(A_host, C_host, D_host, ans_host);
    }

    long long t8(scl_t* A_host, scl_t* B_host, scl_t* C_host, scl_t* D_host, scl_t* ans_host) {
        return t8<i_2d, j_2d, k_2d>(A_host, B_host, C_host, D_host, ans_host);
    }

    long long t9(scl_t* A_host, scl_t* B_host, scl_t* C_host, scl_t* D_host, scl_t* ans_host) {
        return t9<i_2d, j_2d, k_2d>(A_host, B_host, C_host, D_host, ans_host);
    }

    long long t10(scl_t* A_host, scl_t* B_host, scl_t* C_host, scl_t* D_host, scl_t* ans_host) {
        return t10<i_2d, j_2d, k_2d>(A_host, B_host, C_host, D_host, ans_host);
    }
}