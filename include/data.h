#ifndef DATA_H_
#define DATA_H_

#include "sizes.h"
#include <cstdlib>

template <typename T>
class Data
{
public:
    static T *get(int size = Sizes<T>::max_size())
    {
        static T *data = nullptr;
        if (!data)
        {
            data = (T *)aligned_alloc(32, (size * sizeof(T) + 31) / 32 * 32);
            for (int i = 0; i < size; ++i)
                data[i] = rand() % 128;
        }
        return data;
    }
};

#endif // DATA_H_
