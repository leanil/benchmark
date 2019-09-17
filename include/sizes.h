#ifndef SIZES_H_
#define SIZES_H_

#include <algorithm>
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
        max_size(adjusted_sizes.back());
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
        for (int x : get(MUL_OF))
            bench->Arg(x);
    }
};

#endif // SIZES_H_
