#pragma once
extern "C" {
#include "futhark_test.h"
}
#include "util.h"
#include <chrono>
#include <functional>

using namespace std;

namespace futhark {

    auto config = futhark_context_config_new();
    auto ctx = futhark_context_new(config);

    void close_test() {
        futhark_context_clear_caches(ctx);
        futhark_context_free(ctx);
        futhark_context_config_free(config);
    }
    
    /****** baseline ******/

    template<int i>
    long long baseline_sum(scl_t* A_host = nullptr, scl_t* a_host = nullptr) {
        //Context ctx(config);
        auto A = init<i>(A_host, ctx);
        auto a = init_scalar(a_host);
        auto start = chrono::high_resolution_clock::now();
        futhark_entry_sum(ctx, a_host, A.fut);
        futhark_context_sync(ctx);
        auto done = chrono::high_resolution_clock::now();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long baseline_prod(scl_t* A_host = nullptr, scl_t* c_host = nullptr, scl_t* B_host = nullptr) {
        //Context ctx(config);
        auto A = init<i>(A_host, ctx);
        auto c = init_scalar(c_host);
        auto B = init<i>(B_host, ctx, false);
        auto start = chrono::high_resolution_clock::now();
        futhark_entry_prod(ctx, &B.fut, *c_host, A.fut);
        futhark_context_sync(ctx);
        auto done = chrono::high_resolution_clock::now();
        futhark_values_f32_1d(ctx, B.fut, B_host);
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    /****** 1d ******/

    template<int i>
    long long dot(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* a_host = nullptr) {
        //Context ctx(config);
        auto A = init<i>(A_host, ctx);
        auto B = init<i>(B_host, ctx);
        auto a = init_scalar(a_host, false);
        auto start = chrono::high_resolution_clock::now();
        futhark_context_sync(ctx);
        futhark_entry_dot(ctx, a_host, A.fut, B.fut);
        auto done = chrono::high_resolution_clock::now();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long dot1(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* a_host = nullptr) {
        //Context ctx(config);
        auto A = init<i>(A_host, ctx);
        auto B = init<i>(B_host, ctx);
        auto C = init<i>(C_host, ctx);
        auto a = init_scalar(a_host, false);
        auto start = chrono::high_resolution_clock::now();
        futhark_entry_dot1(ctx, a_host, A.fut, B.fut, C.fut);
        futhark_context_sync(ctx);
        auto done = chrono::high_resolution_clock::now();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long dot2(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* a_host = nullptr) {
        //Context ctx(config);
        auto A = init<i>(A_host, ctx);
        auto B = init<i>(B_host, ctx);
        auto C = init<i>(C_host, ctx);
        auto a = init_scalar(a_host, false);
        auto start = chrono::high_resolution_clock::now();
        futhark_entry_dot2(ctx, a_host, A.fut, B.fut, C.fut);
        futhark_context_sync(ctx);
        auto done = chrono::high_resolution_clock::now();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long dot3(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* a_host = nullptr) {
        //Context ctx(config);
        auto A = init<i>(A_host, ctx);
        auto B = init<i>(B_host, ctx);
        auto C = init<i>(C_host, ctx);
        auto a = init_scalar(a_host, false);
        auto start = chrono::high_resolution_clock::now();
        futhark_entry_dot3(ctx, a_host, A.fut, B.fut, C.fut);
        futhark_context_sync(ctx);
        auto done = chrono::high_resolution_clock::now();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long dot4(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* a_host = nullptr) {
        //Context ctx(config);
        auto A = init<i>(A_host, ctx);
        auto B = init<i>(B_host, ctx);
        auto C = init<i>(C_host, ctx);
        auto D = init<i>(D_host, ctx);
        auto a = init_scalar(a_host, false);
        auto start = chrono::high_resolution_clock::now();
        futhark_entry_dot4(ctx, a_host, A.fut, B.fut, C.fut, D.fut);
        futhark_context_sync(ctx);
        auto done = chrono::high_resolution_clock::now();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long dot5(scl_t* a = nullptr, scl_t* A_host = nullptr, scl_t* b = nullptr, scl_t* B_host = nullptr, scl_t* c = nullptr, scl_t* C_host = nullptr, scl_t* d = nullptr, scl_t* D_host = nullptr, scl_t* sum_host = nullptr) {
        //Context ctx(config);
        auto _a = init_scalar(a);
        auto A = init<i>(A_host, ctx);
        auto _b = init_scalar(b);
        auto B = init<i>(B_host, ctx);
        auto _c = init_scalar(c);
        auto C = init<i>(C_host, ctx);
        auto _d = init_scalar(d);
        auto D = init<i>(D_host, ctx);
        auto sum = init_scalar(sum_host, false);
        auto start = chrono::high_resolution_clock::now();
        futhark_entry_dot5(ctx, sum_host, *a, A.fut, *b, B.fut, *c, C.fut, *d, D.fut);
        futhark_context_sync(ctx);
        auto done = chrono::high_resolution_clock::now();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long dot6(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* a_host = nullptr) {
        //Context ctx(config);
        auto A = init<i>(A_host, ctx);
        auto B = init<i>(B_host, ctx);
        auto C = init<i>(C_host, ctx);
        auto D = init<i>(D_host, ctx);
        auto a = init_scalar(a_host, false);
        auto start = chrono::high_resolution_clock::now();
        futhark_entry_dot6(ctx, a_host, A.fut, B.fut, C.fut, D.fut);
        futhark_context_sync(ctx);
        auto done = chrono::high_resolution_clock::now();
        return chrono::duration_cast<time_precision>(done - start).count();
    }
}