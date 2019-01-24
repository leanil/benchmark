#include "baseline.h"
#include "baseline.hpp"

Eigen::QueueInterface queueInterface((cl::sycl::gpu_selector()));
Eigen::SyclDevice sycl_device(&queueInterface);

namespace eigen {

    long long baseline_sum(scl_t* A_host, scl_t* a_host) {
        return baseline_sum<i_1d>(A_host, a_host);
    }

    long long baseline_prod(scl_t* A_host, scl_t* c, scl_t* B_host) {
        return baseline_prod<i_1d>(A_host, c, B_host);
    }

}