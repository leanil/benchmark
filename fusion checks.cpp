//#include "blas.hpp"
#include "eigen.hpp"
#include "naive.hpp"
#include "vectorized.hpp"
#include "utility.h"
#include <fstream>
#include <iostream>

using namespace std;

int main()
{
    cout << make_histogram(tester(BaselineSum(), 1 << 18, 1000), 20);
    cout << make_histogram(tester(BaselineSum(), 1 << 18, 1000), 20);
    /*cout << tester(BaselineSum(), 1 << 18, 500).min_time << endl
        << tester(naive::BaselineSum(), 1 << 18, 500).min_time << endl
        << tester(vectorized::BaselineSum(), 1 << 18, 500).min_time << endl;*/
    /*for (int i = 14; i <= 18; ++i) {
        cout << tester(BaselineProd(), 1 << i, 100).min_time << " "
            << tester(BaselineSum(), 1 << i, 100).min_time << " "
            << tester(Dot(), 1 << i, 100).min_time << " "
            << tester(Dot2(), 1 << i, 100).min_time << " "
            << tester(Dot3(), 1 << i, 100).min_time << " "
            << tester(Dot4(), 1 << i, 100).min_time << " "
            << tester(Dot5(), 1 << i, 100).min_time << " "
            << tester(Dot6(), 1 << i, 100).min_time << endl;
    }
    cout << endl;
    for (int i = 14; i <= 18; ++i) {
        cout << tester(naive::BaselineProd(), 1 << i, 100).min_time << " "
            << tester(naive::BaselineSum(), 1 << i, 100).min_time << " "
            << tester(naive::Dot(), 1 << i, 100).min_time << " "
            << tester(naive::Dot2(), 1 << i, 100).min_time << " "
            << tester(naive::Dot3(), 1 << i, 100).min_time << " "
            << tester(naive::Dot4(), 1 << i, 100).min_time << " "
            << tester(naive::Dot5(), 1 << i, 100).min_time << " "
            << tester(naive::Dot6(), 1 << i, 100).min_time << endl;
    }*/
    /*cout << make_histogram(tester(Dot3(), 1 << 16, 1000), 10)
        << make_histogram(tester(Dot6(), 1 << 16, 1000), 10);*/
    return 0;
}
