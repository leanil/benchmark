#include "baseline.cu"
#include "1d.cu"
#include "2d.cu"
#include "cublas_v2.h"
#include "helper_cuda.h"
//#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

typedef void(*BenchmarkPtr)(cublasHandle_t);

vector<vector<BenchmarkPtr>> bench{ 
    {baseline_sum, baseline_inc, baseline_prod},
    {dot, dot2, dot3, dot4, dot5},
    {t1, t3, t4, t5, t7, t8, t9, t10} };

int main(int argc, char** argv) {
    if(argc != 3) {
        cout << "Usage: cuBLAS_test <benchmark type [0-3]> <benchmark>" << endl;
        return 0;
    }
    int set = atoi(argv[1]), id = atoi(argv[2]);
    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));
    bench[set][id](handle);
    checkCudaErrors(cublasDestroy(handle));
}