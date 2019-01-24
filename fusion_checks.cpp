//#include "blas.hpp"
#include "eigen.hpp"
#include "futhark.hpp"
#include "naive.hpp"
#include "vectorized.hpp"
#include "utility.h"
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

const int min_size = 14, max_size = 18, sizes = max_size - min_size + 1;
const int input_cnt = 4, mode_cnt = 4, task_cnt = 8;

vector<vector<AlignedData>> inputs(max_size + 1);
vector<vector<scl_t*>> ptrs(max_size + 1);

enum TestMode {
    NAIVE,
    VECTORIZED,
    EIGEN,
    FUTHARK
};

std::array<std::array<std::function<TestResult(int, std::vector<scl_t*> const&, int)>, task_cnt>, mode_cnt> testers{
    naive::testers, vectorized::testers, eigen::testers, futhark::testers };

void check_correctness() {
    cout << naive::BaselineSum(1, ptrs[18]).ans << " " << eigen::BaselineSum(1, ptrs[18]).ans << " " << vectorized::BaselineSum(1, ptrs[18]).ans << " " << futhark::BaselineSum(1, ptrs[18]).ans << endl;
    cout << naive::Dot(1 << 18, ptrs[18]).ans << " " << eigen::Dot(1 << 18, ptrs[18]).ans << " " << vectorized::Dot(1 << 18, ptrs[18]).ans << " " << futhark::Dot(1 << 18, ptrs[18]).ans << endl;
    cout << naive::Dot1(1 << 18, ptrs[18]).ans << " " << eigen::Dot1(1 << 18, ptrs[18]).ans << " " << vectorized::Dot1(1 << 18, ptrs[18]).ans << " " << futhark::Dot1(1 << 18, ptrs[18]).ans << endl;
    cout << naive::Dot2(1 << 18, ptrs[18]).ans << " " << eigen::Dot2(1 << 18, ptrs[18]).ans << " " << vectorized::Dot2(1 << 18, ptrs[18]).ans << " " << futhark::Dot2(1 << 18, ptrs[18]).ans << endl;
    cout << naive::Dot3(1 << 18, ptrs[18]).ans << " " << eigen::Dot3(1 << 18, ptrs[18]).ans << " " << vectorized::Dot3(1 << 18, ptrs[18]).ans << " " << futhark::Dot3(1 << 18, ptrs[18]).ans << endl;
    cout << naive::Dot4(1 << 18, ptrs[18]).ans << " " << eigen::Dot4(1 << 18, ptrs[18]).ans << " " << vectorized::Dot4(1 << 18, ptrs[18]).ans << " " << futhark::Dot4(1 << 18, ptrs[18]).ans << endl;
    cout << naive::Dot5(1 << 18, ptrs[18]).ans << " " << eigen::Dot5(1 << 18, ptrs[18]).ans << " " << vectorized::Dot5(1 << 18, ptrs[18]).ans << " " << futhark::Dot5(1 << 18, ptrs[18]).ans << endl;
    cout << naive::Dot6(1 << 18, ptrs[18]).ans << " " << eigen::Dot6(1 << 18, ptrs[18]).ans << " " << vectorized::Dot6(1 << 18, ptrs[18]).ans << " " << futhark::Dot6(1 << 18, ptrs[18]).ans << endl;
}

void time_matrix(int test_cnt) {
    for (int size = min_size; size <= max_size; ++size) {
        for (int task = 0; task < task_cnt; ++task) {
            cout << '(';
            for (int mode = 0; mode < mode_cnt; ++mode)
                cout << (mode ? "," : "") << testers[mode][task](1 << size, ptrs[size], test_cnt).min_time;
            (cout << ") ").flush();
        }
        cout << endl;
    }
}

int main()
{
    for (int i = min_size; i <= max_size; ++i) {
        for (int j = 0; j < input_cnt; ++j) {
            inputs[i].emplace_back(1 << i, 16);
            ptrs[i].push_back(inputs[i].back().data);
        }
    }
    //check_correctness();
    //cout << endl;
    time_matrix(100);
    force_cache_clear();

    //cout << tester(eigen::BaselineSum(1 << 18, ptrs[18]), 500).min_time << endl
    //    << tester(naive::BaselineSum(1 << 18, ptrs[18]), 500).min_time << endl
    //    << tester(vectorized::BaselineSum(1 << 18, ptrs[18]), 500).min_time << endl
    //    << tester(futhark::BaselineSum(1 << 18, ptrs[18]), 500).min_time << endl;


    /*cout << make_histogram(tester(Dot3(), 1 << 16, 1000), 10)
        << make_histogram(tester(Dot6(), 1 << 16, 1000), 10);*/
#ifdef _WIN32
    cin.ignore();
#endif
}
