#ifndef SIZES_H_
#define SIZES_H_

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <benchmark/benchmark.h>

template <typename T>
class Sizes
{
public:
    static const std::vector<int> &get(int mul_of = 1) // sizes must be a multiple of this
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
            {
                s /= sizeof(T);
                sizes.push_back(s + (s % mul_of ? mul_of - s % mul_of : 0));
            }
        }
        return sizes;
    }
    template <int BENCH_CNT, int MUL_OF>
    static void set(benchmark::internal::Benchmark *bench)
    {
        for (int b = 0; b < BENCH_CNT; ++b)
            for (int x : get(MUL_OF))
                bench->Args({b, x});
    }
};

#endif // SIZES_H_
