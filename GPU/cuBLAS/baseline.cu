#include "util.h"
#include "cublas_v2.h"
#include "helper_cuda.h"
#include <iostream>

using namespace std;

void baseline_sum(cublasHandle_t handle) {
    auto A = init<i_1d>();
    scl_t a;
    checkCudaErrors(cublasSasum(handle, i_1d, A, 1, &a));
    cout << a << endl;
}

void baseline_inc(cublasHandle_t handle) {
    auto A = init<i_1d>();
    auto c1 = init<i_1d>(CONST, 1);
    scl_t alpha = 1;
    checkCudaErrors(cublasSaxpy(handle, i_1d, &alpha, c1, 1, A, 1));
    cout << *A.tmp_data << ' ';
    checkCudaErrors(cublasGetVector(i_1d, sizeof(scl_t), A.gpu_data, 1, A.tmp_data, 1));
    cout << *A.tmp_data << endl;
}

void baseline_prod(cublasHandle_t handle) {
    auto A = init<i_1d>();
    scl_t c = 3.141592;
    checkCudaErrors(cublasSscal(handle, i_1d, &c, A, 1));
    checkCudaErrors(cublasGetVector(i_1d, sizeof(scl_t), A.gpu_data, 1, A.tmp_data, 1));
    cout << *A.tmp_data << endl;
}
