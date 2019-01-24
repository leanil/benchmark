#pragma once
#include "../config.h"

namespace eigen {

    long long dot(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* a_host = nullptr);
    long long dot1(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* a_host = nullptr);
    long long dot2(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* a_host = nullptr);
    long long dot3(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* a_host = nullptr);
    long long dot4(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* a_host = nullptr);
    long long dot5(scl_t* a = nullptr, scl_t* A_host = nullptr, scl_t* b = nullptr, scl_t* B_host = nullptr, scl_t* c = nullptr, scl_t* C_host = nullptr, scl_t* d = nullptr, scl_t* D_host = nullptr, scl_t* sum_host = nullptr);
    long long dot6(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* a_host = nullptr);

}