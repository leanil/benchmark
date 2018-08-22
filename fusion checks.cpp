//#include "blas.hpp"
#include "eigen.hpp"
#include "utility.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>

#include <type_traits>
#include <vector>

using namespace std;

int main()
{
    for (int i = 14; i <= 18; ++i) {
        cout << tester(BaselineProd(), 1 << i, 100).min_time << " "
            << tester(BaselineSum(), 1 << i, 100).min_time << " "
            << tester(Dot(), 1 << i, 100).min_time << " "
            << tester(Dot2(), 1 << i, 100).min_time << " "
            << tester(Dot3(), 1 << i, 100).min_time << " "
            << tester(Dot4(), 1 << i, 100).min_time << " "
            << tester(Dot5(), 1 << i, 100).min_time << " "
            << tester(Dot6(), 1 << i, 100).min_time << endl;
    }
    cout << make_histogram(tester(Dot3(), 1 << 16, 1000), 10)
        << make_histogram(tester(Dot6(), 1 << 16, 1000), 10);
    system("PAUSE");
    return 0;
}
