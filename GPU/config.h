#pragma once
#include <chrono>
#include <memory>
#include <random>

using scl_t = float;
using time_precision = std::chrono::microseconds;

constexpr int i_1d = 10000000, i_2d = 1000, j_2d = 1000, k_2d = 1000;

template<int...>
struct Prod;

template<int H, int... T>
struct Prod<H, T...> {
    static constexpr int value = H * Prod<T...>::value;
};

template<>
struct Prod<> {
    static constexpr int value = 1;
};

scl_t* allocate(int size, bool randomize = true);

template<int... Dims>
scl_t* allocate(bool randomize = true) {
    return allocate(Prod<Dims...>::value, randomize);
}

std::unique_ptr<scl_t> init_scalar(scl_t*& p, bool randomize = true);

struct Storage {
    scl_t* tmp_data = nullptr;
    ~Storage() {
        if (tmp_data)
            delete[] tmp_data;
    }
};