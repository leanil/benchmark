#pragma once
#include "util.h"
#include <chrono>

using namespace std;

namespace eigen {

    template<int i, int j>
    long long t1(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* ans_host = nullptr) {
        auto A = init<i, j>(A_host);
        auto B = init<j, 1>(B_host);
        auto ans = init<i, 1>(ans_host, false);
        auto start = chrono::high_resolution_clock::now();
        ans().device(sycl_device) = A().contract(B(), Dimensions<1>{ { {1, 0} }});
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(ans_host, ans.gpu_data, i * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i, int j>
    long long t2(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* ans_host = nullptr) {
        auto A = init<i, j>(A_host);
        auto B = init<j, 1>(B_host);
        auto C = init<i, 1>(C_host);
        auto ans = init<i, 1>(ans_host, false);
        auto start = chrono::high_resolution_clock::now();
        ans().device(sycl_device) = A().contract(B(), Dimensions<1>{ { {1, 0} }}) * C();
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(ans_host, ans.gpu_data, i * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i, int j>
    long long t3(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* ans_host = nullptr) {
        auto A = init<i, j>(A_host);
        auto B = init<i, j>(B_host);
        auto C = init<j, 1>(C_host);
        auto ans = init<i, 1>(ans_host, false);
        auto start = chrono::high_resolution_clock::now();
        ans().device(sycl_device) = (A() + B()).contract(C(), Dimensions<1>{ { {1, 0} }});
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(ans_host, ans.gpu_data, i * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i, int j>
    long long t4(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr) {
        auto A = init<i, j>(A_host);
        auto B = init<i, j>(B_host);
        auto C = init<j, 1>(C_host);
        auto D = init<j, 1>(D_host);
        auto ans = init<i, 1>(ans_host, false);
        auto start = chrono::high_resolution_clock::now();
        ans().device(sycl_device) = (A() + B()).contract((C() + D()), Dimensions<1>{ { {1, 0} }});
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(ans_host, ans.gpu_data, i * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i, int j>
    long long t5(scl_t* a = nullptr, scl_t* A_host = nullptr, scl_t* b = nullptr, scl_t* B_host = nullptr, scl_t* c = nullptr, scl_t* C_host = nullptr, scl_t* d = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr) {
        auto _a = init_scalar(a);
        auto A = init<i, j>(A_host);
        auto _b = init_scalar(b);
        auto B = init<i, j>(B_host);
        auto _c = init_scalar(c);
        auto C = init<j, 1>(C_host);
        auto _d = init_scalar(d);
        auto D = init<j, 1>(D_host);
        auto ans = init<i, 1>(ans_host, false);
        auto start = chrono::high_resolution_clock::now();
        ans().device(sycl_device) = (*a*A() + *b*B()).contract((*c*C() + *d*D()), Dimensions<1>{ { {1, 0} }});
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(ans_host, ans.gpu_data, i * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i, int j>
    long long t6(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr) {
        auto A = init<i>(A_host);
        auto B = init<j>(B_host);
        auto C = init<i>(C_host);
        auto D = init<j>(D_host);
        auto ans = init<i>(ans_host, false);
        auto start = chrono::high_resolution_clock::now();
        auto con_dims = Dimensions<0>{};
        ans().device(sycl_device) = (A().contract(B(), con_dims)*C().contract(D(), con_dims)).sum(Eigen::array<int, 1>{ 1 });
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(ans_host, ans.gpu_data, i * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i, int j, int k>
    long long t7(scl_t* A_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr) {
        auto A = init<i, j>(A_host);
        auto C = init<j, k>(C_host);
        auto D = init<k, 1>(D_host);
        auto ans = init<i, 1>(ans_host, false);
        auto start = chrono::high_resolution_clock::now();
        auto con_dims = Dimensions<1>{ { {1, 0} } };
        ans().device(sycl_device) = A().contract(C().contract(D(), con_dims), con_dims);
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(ans_host, ans.gpu_data, i * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i, int j, int k>
    long long t8(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr) {
        auto A = init<i, j>(A_host);
        auto B = init<i, j>(B_host);
        auto C = init<j, k>(C_host);
        auto D = init<k, 1>(D_host);
        auto ans = init<i, 1>(ans_host, false);
        auto start = chrono::high_resolution_clock::now();
        auto con_dims = Dimensions<1>{ { {1, 0} } };
        ans().device(sycl_device) = (A() + B()).contract(C().contract(D(), con_dims), con_dims);
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(ans_host, ans.gpu_data, i * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i, int j, int k>
    long long t9(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr) {
        auto A = init<i, k>(A_host);
        auto B = init<k, j>(B_host);
        auto C = init<j, 1>(C_host);
        auto D = init<j, 1>(D_host);
        auto ans = init<i, 1>(ans_host, false);
        auto start = chrono::high_resolution_clock::now();
        auto con_dims = Dimensions<1>{ { {1, 0} } };
        ans().device(sycl_device) = A().contract(B(), con_dims).contract(C() + D(), con_dims);
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(ans_host, ans.gpu_data, i * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

    template<int i, int j, int k>
    long long t10(scl_t* A_host = nullptr, scl_t* B_host = nullptr, scl_t* C_host = nullptr, scl_t* D_host = nullptr, scl_t* ans_host = nullptr) {
        auto A = init<i, k>(A_host);
        auto B = init<k, j>(B_host);
        auto C = init<j, k>(C_host);
        auto D = init<k, 1>(D_host);
        auto ans = init<i, 1>(ans_host, false);
        auto start = chrono::high_resolution_clock::now();
        auto con_dims = Dimensions<1>{ { {1, 0} } };
        ans().device(sycl_device) = A().contract(B(), con_dims).contract(C().contract(D(), con_dims), con_dims);
        auto done = chrono::high_resolution_clock::now();
        sycl_device.memcpyDeviceToHost(ans_host, ans.gpu_data, i * sizeof(scl_t));
        sycl_device.synchronize();
        return chrono::duration_cast<time_precision>(done - start).count();
    }

}
