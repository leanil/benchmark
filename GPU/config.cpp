#include "config.h"

scl_t* allocate(int size, bool randomize) {
    scl_t* data = new scl_t[size];
    if (randomize) {
        static std::default_random_engine gen;
        static std::uniform_real_distribution<scl_t> dist;
        for (int i = 0; i < size; ++i)
            data[i] = dist(gen);
    }
    return data;
}

std::unique_ptr<scl_t> init_scalar(scl_t*& p, bool randomize) {
    if (p) return std::unique_ptr<scl_t>();
    p = new scl_t;
    if (randomize) {
        static std::default_random_engine gen;
        static std::uniform_real_distribution<scl_t> dist;
        *p = dist(gen);
    }
    return std::unique_ptr<scl_t>(p);
}