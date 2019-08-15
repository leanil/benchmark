#pragma once
extern "C" {
#include "baseline.h"
}
#include "util.h"
#include <chrono>
#include <functional>

using namespace std;

namespace futhark {

    template<int i>
    long long baseline_sum(scl_t* A_host = nullptr, scl_t* a_host = nullptr) {
        auto config = futhark_context_config_new();
        auto context = futhark_context_new(config);
        auto A = init<i>(A_host, bind(futhark_new_f32_1d,context, placeholders::_1, placeholders::_2), bind(futhark_free_f32_1d, context, placeholders::_1));
        auto a = init_scalar(a_host);
        auto start = chrono::high_resolution_clock::now();
        futhark_entry_sum(context, a_host, A.fut);
        auto done = chrono::high_resolution_clock::now();
        futhark_context_free(context);
        futhark_context_config_free(config);
        return chrono::duration_cast<time_precision>(done - start).count();
    }
}