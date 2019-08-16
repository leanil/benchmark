#ifndef DATA_H_
#define DATA_H_

#include "sizes.h"
#include <cstdlib>

template <typename T>
class Data
{
    static T *data;

public:
    static T *get(int size)
    {
        if (!data)
        {
            data = (T *)aligned_alloc(32, size * sizeof(T));
            for (int i = 0; i < size; ++i)
                data[i] = rand() % 128;
        }
        return data;
    }
    static T *get()
    {
        auto sizes = Sizes<T>::get();
        return get(*max_element(sizes.begin(), sizes.end()));
    }
};

template <typename T>
T *Data<T>::data = nullptr;

#endif // DATA_H_
