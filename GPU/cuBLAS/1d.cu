#include "util.h"
#include "cublas_v2.h"
#include "helper_cuda.h"
#include <iostream>

using namespace std;

void dot(cublasHandle_t handle) {
    auto A = init<i_1d>();
    auto B = init<i_1d>();
    scl_t a;
    checkCudaErrors(cublasSdot(handle, i_1d, A, 1, B, 1, &a));
    cout << a << endl;
}

void dot2(cublasHandle_t handle) {
    auto A = init<i_1d>();
    auto B = init<i_1d>();
    auto C = init<i_1d>();
    scl_t a, alpha = 1;
    checkCudaErrors(cublasSaxpy(handle, i_1d, &alpha, A, 1, B, 1));
    checkCudaErrors(cublasSdot(handle, i_1d, B, 1, C, 1, &a));
    cout << a << endl;
}

void dot3(cublasHandle_t handle) {
    auto A = init<i_1d>();
    auto B = init<i_1d>();
    auto C = init<i_1d>();
    scl_t a, alpha = 1;
    checkCudaErrors(cublasSaxpy(handle, i_1d, &alpha, A, 1, B, 1));
    alpha = -1;
    checkCudaErrors(cublasSaxpy(handle, i_1d, &alpha, C, 1, A, 1));
    checkCudaErrors(cublasSdot(handle, i_1d, B, 1, A, 1, &a));
    cout << a << endl;
}

void dot4(cublasHandle_t handle) {
    auto A = init<i_1d>();
    auto B = init<i_1d>();
    auto C = init<i_1d>();
    auto D = init<i_1d>();
    scl_t a, alpha = 1;
    checkCudaErrors(cublasSaxpy(handle, i_1d, &alpha, A, 1, B, 1));
    alpha = -1;
    checkCudaErrors(cublasSaxpy(handle, i_1d, &alpha, D, 1, C, 1));
    checkCudaErrors(cublasSdot(handle, i_1d, B, 1, C, 1, &a));
    cout << a << endl;
}

void dot5(cublasHandle_t handle) {
    auto A = init<i_1d>();
    auto B = init<i_1d>();
    auto C = init<i_1d>();
    auto D = init<i_1d>();
    scl_t a = 1.2, b = 3.4, c = 5.6, d = 7.8;
    scl_t result, alpha = 1;
    checkCudaErrors(cublasSscal(handle, i_1d, &a, A, 1));
    checkCudaErrors(cublasSscal(handle, i_1d, &b, B, 1));
    checkCudaErrors(cublasSscal(handle, i_1d, &c, C, 1));
    checkCudaErrors(cublasSscal(handle, i_1d, &d, D, 1));
    checkCudaErrors(cublasSaxpy(handle, i_1d, &alpha, A, 1, B, 1));
    checkCudaErrors(cublasSaxpy(handle, i_1d, &alpha, C, 1, D, 1));
    checkCudaErrors(cublasSdot(handle, i_1d, B, 1, D, 1, &result));
    cout << result << endl;
}