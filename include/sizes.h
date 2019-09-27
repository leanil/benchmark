#ifndef SIZES_H_
#define SIZES_H_

#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <benchmark/benchmark.h>

template <typename T>
class Sizes
{
public:
    static std::vector<int> get(int mul_of = 1) // sizes must be a multiple of this
    {
        static std::vector<int> sizes;
        if (sizes.empty())
        {
            std::cerr << "data sizes (Bytes):" << std::endl;
            std::string line;
            std::getline(std::cin, line);
            std::stringstream ss(line);
            int s;
            while (ss >> s)
                sizes.push_back(s / sizeof(T));
            std::sort(sizes.begin(), sizes.end());
        }
        std::vector<int> adjusted_sizes(sizes.size());
        std::transform(sizes.begin(), sizes.end(), adjusted_sizes.begin(),
                       [&](int x) { return (x + mul_of - 1) / mul_of * mul_of; });
        adjusted_sizes.erase(std::unique(adjusted_sizes.begin(), adjusted_sizes.end()), adjusted_sizes.end());
        return adjusted_sizes;
    }
    static int max_size(int new_size = 0)
    {
        static int ms = 0;
        return ms = std::max(ms, new_size);
    }
    template <int MUL_OF>
    static void set(benchmark::internal::Benchmark *bench)
    {
        auto sizes = get(MUL_OF);
        for (int x : sizes)
            bench->Arg(x);
        max_size(sizes.back());
    }
    template <int MUL_OF>
    static void set_mat(benchmark::internal::Benchmark *bench)
    {
        auto sizes = get(MUL_OF);
        static const std::array fixed_sizes{120, 360, 600};
        for (int x : fixed_sizes)
            for (int y : sizes)
                bench->Args({x, y});
        max_size(fixed_sizes.back() * sizes.back());
    }
};

#endif // SIZES_H_
