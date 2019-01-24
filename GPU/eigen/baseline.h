#pragma once
#include "../config.h"

namespace eigen {

    long long baseline_sum(scl_t* A_host = nullptr, scl_t* a_host = nullptr);
    long long baseline_prod(scl_t* A_host = nullptr, scl_t* c = nullptr, scl_t* B_host = nullptr);

}