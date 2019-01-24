#pragma once

#include "../config.h"
#define EIGEN_USE_SYCL
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <random>

constexpr auto DataLayout = Eigen::RowMajor;

template<int rank>
using Tensor = Eigen::Tensor<scl_t, rank, DataLayout>;

template<int rank>
using TensorMap = Eigen::TensorMap<Tensor<rank>>;

template<int rank>
using Dimensions = Eigen::array<Eigen::IndexPair<int>, rank>;

extern Eigen::QueueInterface queueInterface;
extern Eigen::SyclDevice sycl_device;

template<int rank>
struct EigenStorage : Storage {
    EigenStorage(scl_t* gpu_data, TensorMap<rank> tensor) : gpu_data{ gpu_data }, tensor{ tensor } {}
    scl_t* gpu_data;
    TensorMap<rank> tensor;
    TensorMap<rank> operator()() const { return tensor; }
    ~EigenStorage() {
        sycl_device.deallocate(gpu_data);
    }
};

template<int... Dims>
auto init(scl_t*& p, bool is_input = true) {
    constexpr int rank = sizeof...(Dims), size = Prod<Dims...>::value;
    scl_t* gpu_data = static_cast<scl_t*>(sycl_device.allocate(size * sizeof(scl_t)));
    EigenStorage<rank> s(gpu_data, TensorMap<rank>(gpu_data, Dims...));
    if (!p)
        s.tmp_data = p = allocate(size, is_input);
    sycl_device.memcpyHostToDevice(s.gpu_data, p, size * sizeof(scl_t));
    return s;
}
