#pragma once

#include "../config.h"
#include "cublas_v2.h"
#include "helper_cuda.h"
#include <algorithm>
#include <memory>
#include <random>

struct CudaStorage : Storage {
    scl_t* gpu_data;
    operator scl_t*() const { return gpu_data; }
    ~CudaStorage() {
        checkCudaErrors(cudaFree(gpu_data));
    }
};

enum InitMem {
    RANDOMIZE, CONST, NONE
};

template<int... Dims>
auto init(InitMem init = RANDOMIZE, scl_t val = 0) {
    constexpr int size = Prod<Dims...>::value;
    CudaStorage storage;
    storage.tmp_data = allocate(size, init == RANDOMIZE);
    if (init == CONST)
        std::fill(storage.tmp_data, storage.tmp_data + size, val);
    checkCudaErrors(cudaMalloc((void**)&storage.gpu_data, size * sizeof(scl_t)));
    if (init != NONE)
        checkCudaErrors(cublasSetVector(size, sizeof(scl_t), storage.tmp_data, 1, storage.gpu_data, 1));
    return storage;
}
