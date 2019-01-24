#pragma once
#include "util.h"
#include <chrono>

using namespace std;

namespace eigen {

    template<int i>
    long long dot(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* a_host = nullptr) {
        auto A = init<i>(A_host);
        auto B = init<i>(B_host);
        auto a = init<>(a_host, false);
        auto start = chrono::high_resolution_clock::now();
        a().device(sycl_device) = (A()*B()).sum();
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(a_host, a.gpu_data, 1 * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long dot1(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* a_host = nullptr) {
        auto A = init<i>(A_host);
        auto B = init<i>(B_host);
        auto C = init<i>(C_host);
        auto a = init<>(a_host, false);
        auto start = chrono::high_resolution_clock::now();
        a().device(sycl_device) = (A()*B()*C()).sum();
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(a_host, a.gpu_data, 1 * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long dot2(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* a_host = nullptr) {
        auto A = init<i>(A_host);
        auto B = init<i>(B_host);
        auto C = init<i>(C_host);
        auto a = init<>(a_host, false);
        auto start = chrono::high_resolution_clock::now();
        a().device(sycl_device) = ((A() + B())*C()).sum();
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(a_host, a.gpu_data, 1 * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long dot3(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* a_host = nullptr) {
        auto A = init<i>(A_host);
        auto B = init<i>(B_host);
        auto C = init<i>(C_host);
        auto a = init<>(a_host, false);
        auto start = chrono::high_resolution_clock::now();
        a().device(sycl_device) = ((A() + B())*(A() - C())).sum();
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(a_host, a.gpu_data, 1 * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long dot4(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* a_host = nullptr) {
        auto A = init<i>(A_host);
        auto B = init<i>(B_host);
        auto C = init<i>(C_host);
        auto D = init<i>(D_host);
        auto a = init<>(a_host, false);
        auto start = chrono::high_resolution_clock::now();
        a().device(sycl_device) = ((A() + B())*(C() - D())).sum();
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(a_host, a.gpu_data, 1 * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long dot5(scl_t* a = nullptr, scl_t* A_host = nullptr, scl_t* b = nullptr, scl_t* B_host = nullptr, scl_t* c = nullptr, scl_t* C_host = nullptr, scl_t* d = nullptr, scl_t* D_host = nullptr, scl_t* sum_host = nullptr) {
        auto _a = init_scalar(a);
        auto A = init<i>(A_host);
        auto _b = init_scalar(b);
        auto B = init<i>(B_host);
        auto _c = init_scalar(c);
        auto C = init<i>(C_host);
        auto _d = init_scalar(d);
        auto D = init<i>(D_host);
        auto sum = init<>(sum_host, false);
        auto start = chrono::high_resolution_clock::now();
        sum().device(sycl_device) = ((A()**a + B()**b)*(C()**c + D()**d)).sum();
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(sum_host, sum.gpu_data, 1 * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i>
    long long dot6(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* a_host = nullptr) {
        auto A = init<i>(A_host);
        auto B = init<i>(B_host);
        auto C = init<i>(C_host);
        auto D = init<i>(D_host);
        auto a = init<>(a_host, false);
        auto start = chrono::high_resolution_clock::now();
        a().device(sycl_device) = ((A() + B()).eval()*(C() - D()).eval()).sum();
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(a_host, a.gpu_data, 1 * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

}
