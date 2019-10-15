#ifndef SIZES_H_
#define SIZES_H_

#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <benchmark/benchmark.h>

class Sizes
{
public:
    static std::vector<int> get(int id, int batch = 1) // sizes must be multiples of this
    {
        static std::vector<std::vector<int>> sizes;
        if ((int)sizes.size() <= id)
        {
            sizes.resize(id + 1);
            std::cerr << "sizes:" << std::endl;
            std::string line;
            std::getline(std::cin, line);
            std::stringstream ss(line);
            int s;
            while (ss >> s)
                sizes[id].push_back(s);
            std::sort(sizes[id].begin(), sizes[id].end());
        }
        if (batch == 1)
            return sizes[id];
        std::vector<int> adjusted_sizes(sizes[id].size());
        std::transform(sizes[id].begin(), sizes[id].end(), adjusted_sizes.begin(),
                       [&](int x) { return (x + batch - 1) / batch * batch; });
        adjusted_sizes.erase(std::unique(adjusted_sizes.begin(), adjusted_sizes.end()), adjusted_sizes.end());
        return adjusted_sizes;
    }
    static int set(benchmark::internal::Benchmark *bench, int batch)
    {
        auto sizes = get(0, batch);
        for (int x : sizes)
            bench->Arg(x);
        return sizes.back();
    }
    static int set_mat(benchmark::internal::Benchmark *bench, int batch_0, int batch_1)
    {
        auto sizes_0 = get(0, batch_0), sizes_1 = Sizes::get(1, batch_1);
        for (int a : sizes_0)
            for (int b : sizes_1)
                bench->Args({a, b});
        return sizes_0.back() * sizes_1.back();
    }
};

#endif // SIZES_H_
