#ifndef DATA_H_
#define DATA_H_

#include "sizes.h"
#include <cstdlib>
#include <vector>

template <typename T>
class Data
{
public:
    static T *get(int id = 0)
    {
        static std::vector<T *> data;
        while (data.size() <= id)
        {
            int size = max_size(id);
            data.push_back((T *)aligned_alloc(32, (size * sizeof(T) + 31) / 32 * 32));
            for (int i = 0; i < size; ++i)
                data.back()[i] = rand() % 128;
        }
        return data[id];
    }

    static int max_size(int id, int new_size = 0)
    {
        static std::vector<int> max_sizes;
        if (max_sizes.size() <= id)
            max_sizes.resize(id + 1, 0);
        return max_sizes[id] = std::max(max_sizes[id], new_size);
    }
};

#endif // DATA_H_
