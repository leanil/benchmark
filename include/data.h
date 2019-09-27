#ifndef DATA_H_
#define DATA_H_

#include "sizes.h"
#include <cstdlib>
#include <vector>

template <typename T>
class Data
{
public:
    static T *get(int id = 0, int size = Sizes<T>::max_size())
    {
        static std::vector<T*> data;
        while (data.size() <= id)
        {
            data.push_back((T *)aligned_alloc(32, (size * sizeof(T) + 31) / 32 * 32));
            for (int i = 0; i < size; ++i)
                data.back()[i] = rand() % 128;
        }
        return data[id];
    }
};

#endif // DATA_H_
