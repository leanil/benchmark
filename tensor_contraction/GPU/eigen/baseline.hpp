#pragma once
#include "util.h"
#include <chrono>
#include <iostream>
using namespace std;

namespace eigen {

    template<int i>
    long long baseline_sum(scl_t* A_host = nullptr, scl_t* a_host = nullptr) {
        auto A = init<i>(A_host);
        auto a = init<>(a_host, false);
        auto start = chrono::high_resolution_clock::now();
        a().device(sycl_device) = A().sum(); // why is this blocking?
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(a_host, a.gpu_data, 1 * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long baseline_inc(scl_t* A_host = nullptr, scl_t* B_host = nullptr) {
        auto A = init<i>(A_host);
        auto B = init<i>(B_host, false);
        auto start = chrono::high_resolution_clock::now();
        B().device(sycl_device) = A() + A().constant(1);
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(B_host, B.gpu_data, i * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long baseline_prod(scl_t* A_host = nullptr, scl_t* c = nullptr, scl_t* B_host = nullptr) {
        auto A = init<i>(A_host);
        init_scalar(c);
        auto B = init<i>(B_host, false);
        auto start = chrono::high_resolution_clock::now();
        B().device(sycl_device) = A().constant(*c)*A();
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(B_host, B.gpu_data, i * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

}
