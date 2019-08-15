#include "baseline.hpp"
#define EIGEN_USE_SYCL
#include <unsupported/Eigen/CXX11/Tensor>
#include <random>
#include <iostream>

using namespace std;

//using scl_t = float;
//
//constexpr auto DataLayout = Eigen::RowMajor;
//
//template<int rank>
//using Tensor = Eigen::Tensor<scl_t, rank, DataLayout>;
//
//template<int rank>
//using TensorMap = Eigen::TensorMap<Tensor<rank>>;
//
//scl_t* allocate(int size, bool randomize) {
//    scl_t* data = new scl_t[size];
//    if (randomize) {
//        static std::default_random_engine gen;
//        static std::uniform_real_distribution<scl_t> dist;
//        for (int i = 0; i < size; ++i)
//            data[i] = dist(gen);
//    }
//    return data;
//}
//
Eigen::QueueInterface queueInterface((cl::sycl::gpu_selector()));
Eigen::SyclDevice sycl_device(&queueInterface);

scl_t* t1() {
    int i = 100, j = 100;
    scl_t* A_host = allocate(i*j, true);
    scl_t* A_gpu = static_cast<scl_t*>(sycl_device.allocate(i*j * sizeof(scl_t)));
    sycl_device.memcpyHostToDevice(A_gpu, A_host, i*j * sizeof(scl_t));
    TensorMap<2> A(A_gpu, i, j);

    scl_t* B_host = allocate(i*j, true);
    scl_t* B_gpu = static_cast<scl_t*>(sycl_device.allocate(i*j * sizeof(scl_t)));
    sycl_device.memcpyHostToDevice(B_gpu, B_host, i*j * sizeof(scl_t));
    TensorMap<2> B(B_gpu, i, j);

    scl_t* C_host = allocate(i*j, true);
    scl_t* C_gpu = static_cast<scl_t*>(sycl_device.allocate(i *j * sizeof(scl_t)));
    TensorMap<2> C(C_gpu, i, j);

    C.device(sycl_device) = A + B;// A.contract(B, Eigen::array<Eigen::IndexPair<int>, 1>{ { {1, 0} }});
    sycl_device.memcpyDeviceToHost(C_host, C_gpu, i *j * sizeof(scl_t));
    sycl_device.synchronize();
    return C_host;
}

int main() {
    scl_t* ans = t1();
    for (int i = 0; i < 100; ++i)
        cout << ans[i] << ' ';
    cout << eigen::baseline_sum<10000000>() << endl;
    cout << endl;
}