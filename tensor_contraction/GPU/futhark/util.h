#pragma once
#include "../config.h"
#include "futhark_test.h"
#include <type_traits>

struct Context {
    futhark_context* context;
    Context(futhark_context_config* config) : context{ futhark_context_new(config) } {}
    operator futhark_context*() const { return context; }
    ~Context() {
        futhark_context_clear_caches(context);
        futhark_context_free(context);
    }
};

template<unsigned Rank>
struct FutharkArray;

template<>
struct FutharkArray<1> {
    using type = futhark_f32_1d;
    static constexpr auto new_array = futhark_new_f32_1d;
    static constexpr auto free_array = futhark_free_f32_1d;
};

template<typename Fut>
struct FutharkStorage : Storage {
    futhark_context* context;
    typename Fut::type* fut;
    ~FutharkStorage() { Fut::free_array(context, fut); }
};

template<int... Dims>
auto init(scl_t*& p, futhark_context* context, bool is_input = true) {
    using Fut = FutharkArray<sizeof...(Dims)>;
    FutharkStorage<Fut> s;
    s.context = context;
    if (!p)
        s.tmp_data = p = allocate<Dims...>(is_input);
    s.fut = Fut::new_array(context, p, Dims...);
    return s;
}
