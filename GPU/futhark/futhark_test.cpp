#include "futhark_test.hpp"
#include <iostream>

using namespace futhark;
using namespace std;

int main() {
    cout << "warmup: " << baseline_sum<10000000>() << "\n\n";

    cout << "Baseline:\n";
    cout << "sum\n";
    for (int i = 0; i < 10; ++i)
        cout << baseline_sum<10000000>() << endl;
    cout << "\nprod\n";
    for (int i = 0; i < 10; ++i)
        cout << baseline_prod<10000000>() << endl;

    cout << "\n1D:\n";
    cout << "dot\n";
    for (int i = 0; i < 10; ++i)
        cout << dot<10000000>() << endl;
    cout << "\ndot1\n";
    for (int i = 0; i < 10; ++i)
        cout << dot1<10000000>() << endl;
    cout << "\ndot2\n";
    for (int i = 0; i < 10; ++i)
        cout << dot2<10000000>() << endl;
    cout << "\ndot3\n";
    for (int i = 0; i < 10; ++i)
        cout << dot3<10000000>() << endl;
    cout << "\ndot4\n";
    for (int i = 0; i < 10; ++i)
        cout << dot4<10000000>() << endl;
    cout << "\ndot5\n";
    for (int i = 0; i < 10; ++i)
        cout << dot5<10000000>() << endl;
    cout << "\ndot6\n";
    for (int i = 0; i < 10; ++i)
        cout << dot6<10000000>() << endl;

    /*cout << "\n2D:\n";
    cout << "t1\n";
    for (int i = 0; i < 10; ++i)
        cout << t1<1000, 1000>() << endl;*/

    close_test();
}