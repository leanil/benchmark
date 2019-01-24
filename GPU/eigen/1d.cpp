#include "1d.h"
#include "1d.hpp"

namespace eigen {

    long long dot(scl_t* A_host, scl_t* B_host, scl_t* a_host) {
        return dot<i_1d>(A_host, B_host, a_host);
    }

    long long dot1(scl_t* A_host, scl_t* B_host, scl_t* C_host, scl_t* a_host) {
        return dot1<i_1d>(A_host, B_host, C_host, a_host);
    }

    long long dot2(scl_t* A_host, scl_t* B_host, scl_t* C_host, scl_t* a_host) {
        return dot2<i_1d>(A_host, B_host, C_host, a_host);
    }

    long long dot3(scl_t* A_host, scl_t* B_host, scl_t* C_host, scl_t* a_host) {
        return dot3<i_1d>(A_host, B_host, C_host, a_host);
    }

    long long dot4(scl_t* A_host, scl_t* B_host, scl_t* C_host, scl_t* D_host, scl_t* a_host) {
        return dot4<i_1d>(A_host, B_host, C_host, D_host, a_host);
    }

    long long dot5(scl_t* a, scl_t* A_host, scl_t* b, scl_t* B_host, scl_t* c, scl_t* C_host, scl_t* d, scl_t* D_host, scl_t* sum_host) {
        return dot5<i_1d>(a, A_host, b, B_host, c, C_host, d, D_host, sum_host);
    }

    long long dot6(scl_t* A_host, scl_t* B_host, scl_t* C_host, scl_t* D_host, scl_t* a_host) {
        return dot6<i_1d>(A_host, B_host, C_host, D_host, a_host);
    }
}
