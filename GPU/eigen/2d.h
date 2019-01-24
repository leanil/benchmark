#pragma once
#include "../config.h"

namespace eigen {

    long long t1(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* ans_host = nullptr);
    long long t2(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* ans_host = nullptr);
    long long t3(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* ans_host = nullptr);
    long long t4(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr);
    long long t5(scl_t* a = nullptr, scl_t* A_host = nullptr, scl_t* b = nullptr, scl_t* B_host = nullptr, scl_t* c = nullptr, scl_t* C_host = nullptr, scl_t* d = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr);
    long long t6(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr);
    long long t7(scl_t* A_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr);
    long long t8(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr);
    long long t9(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr);
    long long t10(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr);

}