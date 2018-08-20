//#include "blas.hpp"
#include "eigen.hpp"
#include "utility.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>

#include <type_traits>
#include <vector>

using namespace std;

int main()
{
    for (int i = 25; i <= 28; ++i) {
        cout << tester(BaselineSum(), 1 << i, 100).min_time << " "
            << tester(Dot(), 1 << i, 100).min_time << " "
            << tester(Dot2(), 1 << i, 100).min_time << " "
            << tester(Dot3(), 1 << i, 100).min_time << " "
            << tester(Dot4(), 1 << i, 100).min_time << endl;
    }
    system("PAUSE");
    return 0;
}
