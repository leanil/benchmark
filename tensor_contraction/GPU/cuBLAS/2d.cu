#include "util.h"
#include "cublas_v2.h"
#include "helper_cuda.h"
#include <iostream>

using namespace std;

void t1(cublasHandle_t handle) {
    auto A = init<i_2d,j_2d>();
    auto B = init<j_2d>();
    auto ans = init<i_2d>(NONE);
    scl_t alpha = 1, beta = 0;
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, i_2d, j_2d, &alpha, A, i_2d, B, 1, &beta, ans, 1));
    checkCudaErrors(cublasGetVector(i_2d, sizeof(scl_t), ans.gpu_data, 1, ans.tmp_data, 1));
    cout << *ans.tmp_data << endl;
}

void t3(cublasHandle_t handle) {
    auto A = init<i_2d,j_2d>();
    auto B = init<i_2d,j_2d>();
    auto C = init<j_2d>();
    auto ans = init<i_2d>(NONE);
    scl_t alpha = 1, beta = 1;
    checkCudaErrors(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, i_2d, j_2d, &alpha, A, i_2d, &beta, B, i_2d, A, i_2d));
    beta = 0;
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, i_2d, j_2d, &alpha, A, i_2d, B, 1, &beta, ans, 1));
    checkCudaErrors(cublasGetVector(i_2d, sizeof(scl_t), ans.gpu_data, 1, ans.tmp_data, 1));
    cout << *ans.tmp_data << endl;
}

void t4(cublasHandle_t handle) {
    auto A = init<i_2d,j_2d>();
    auto B = init<i_2d,j_2d>();
    auto C = init<j_2d>();
    auto D = init<j_2d>();
    auto ans = init<i_2d>(NONE);
    scl_t alpha = 1, beta = 1;
    checkCudaErrors(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, i_2d, j_2d, &alpha, A, i_2d, &beta, B, i_2d, A, i_2d));
    checkCudaErrors(cublasSaxpy(handle, j_2d, &alpha, C, 1, D, 1));
    beta = 0;
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, i_2d, j_2d, &alpha, A, i_2d, D, 1, &beta, ans, 1));
    checkCudaErrors(cublasGetVector(i_2d, sizeof(scl_t), ans.gpu_data, 1, ans.tmp_data, 1));
    cout << *ans.tmp_data << endl;
}

void t5(cublasHandle_t handle) {
    auto A = init<i_2d,j_2d>();
    auto B = init<i_2d,j_2d>();
    auto C = init<j_2d>();
    auto D = init<j_2d>();
    auto ans = init<i_2d>(NONE);
    scl_t a = 1.2, b = 3.4, c = 5.6, d = 7.8;
    scl_t alpha = 1, beta = 1;
    checkCudaErrors(cublasSscal(handle, i_2d * j_2d, &a, A, 1));
    checkCudaErrors(cublasSscal(handle, i_2d * j_2d, &b, B, 1));
    checkCudaErrors(cublasSscal(handle, j_2d, &c, C, 1));
    checkCudaErrors(cublasSscal(handle, j_2d, &d, D, 1));
    checkCudaErrors(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, i_2d, j_2d, &alpha, A, i_2d, &beta, B, i_2d, A, i_2d));
    checkCudaErrors(cublasSaxpy(handle, j_2d, &alpha, C, 1, D, 1));
    beta = 0;
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, i_2d, j_2d, &alpha, A, i_2d, D, 1, &beta, ans, 1));
    checkCudaErrors(cublasGetVector(i_2d, sizeof(scl_t), ans.gpu_data, 1, ans.tmp_data, 1));
    cout << *ans.tmp_data << endl;
}

void t7(cublasHandle_t handle) {
    auto A = init<i_2d,j_2d>();
    auto C = init<j_2d,k_2d>();
    auto D = init<k_2d>();
    auto ans = init<i_2d>(NONE);
    auto tmp = init<j_2d>(NONE);
    scl_t alpha = 1, beta = 0;
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, j_2d, k_2d, &alpha, C, j_2d, D, 1, &beta, tmp, 1));
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, i_2d, j_2d, &alpha, A, i_2d, tmp, 1, &beta, ans, 1));
    checkCudaErrors(cublasGetVector(i_2d, sizeof(scl_t), ans.gpu_data, 1, ans.tmp_data, 1));
    cout << *ans.tmp_data << endl;
}

void t8(cublasHandle_t handle) {
    auto A = init<i_2d,j_2d>();
    auto B = init<i_2d,j_2d>();
    auto C = init<j_2d,k_2d>();
    auto D = init<k_2d>();
    auto ans = init<i_2d>(NONE);
    auto tmp = init<j_2d>(NONE);
    scl_t alpha = 1, beta = 1;
    checkCudaErrors(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, i_2d, j_2d, &alpha, A, i_2d, &beta, B, i_2d, A, i_2d));
    beta = 0;
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, j_2d, k_2d, &alpha, C, j_2d, D, 1, &beta, tmp, 1));
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, i_2d, j_2d, &alpha, A, i_2d, tmp, 1, &beta, ans, 1));
    checkCudaErrors(cublasGetVector(i_2d, sizeof(scl_t), ans.gpu_data, 1, ans.tmp_data, 1));
    cout << *ans.tmp_data << endl;
}

void t9(cublasHandle_t handle) {
    auto A = init<i_2d,k_2d>();
    auto B = init<k_2d,j_2d>();
    auto C = init<j_2d>();
    auto D = init<j_2d>();
    auto ans = init<i_2d>(NONE);
    auto tmp = init<i_2d,j_2d>(NONE);
    scl_t alpha = 1, beta = 0;
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i_2d, j_2d, k_2d, &alpha, A, i_2d, B, k_2d, &beta, tmp, i_2d));
    checkCudaErrors(cublasSaxpy(handle, j_2d, &alpha, C, 1, D, 1));
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, i_2d, j_2d, &alpha, tmp, i_2d, D, 1, &beta, ans, 1));
    checkCudaErrors(cublasGetVector(i_2d, sizeof(scl_t), ans.gpu_data, 1, ans.tmp_data, 1));
    cout << *ans.tmp_data << endl;
}

void t10(cublasHandle_t handle) {
    auto A = init<i_2d,k_2d>();
    auto B = init<k_2d,j_2d>();
    auto C = init<j_2d,k_2d>();
    auto D = init<k_2d>();
    auto ans = init<i_2d>(NONE);
    auto tmp1 = init<i_2d,j_2d>(NONE);
    auto tmp2 = init<j_2d>(NONE);
    scl_t alpha = 1, beta = 0;
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i_2d, j_2d, k_2d, &alpha, A, i_2d, B, k_2d, &beta, tmp1, i_2d));
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, j_2d, k_2d, &alpha, C, j_2d, D, 1, &beta, tmp2, 1));
    checkCudaErrors(cublasSgemv(handle, CUBLAS_OP_N, i_2d, j_2d, &alpha, tmp1, i_2d, tmp2, 1, &beta, ans, 1));
    checkCudaErrors(cublasGetVector(i_2d, sizeof(scl_t), ans.gpu_data, 1, ans.tmp_data, 1));
    cout << *ans.tmp_data << endl;
}