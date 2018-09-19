#include "utility.h"
#include <cstdlib>
#include <memory>


using namespace std;

void clear_cache() {
    static volatile int tmp[1 << 24]; //larger than the cache
    for (int i = 0; i < 1 << 24; ++i)
        tmp[i] *= 2;
}

Histogram make_histogram(TestResult const& result, int from, int to, int bin_cnt)
{
    Histogram hist{ from, to, bin_cnt, (double)(to - from) / bin_cnt, vector<int>(bin_cnt,0) };
    for (int x : result.times)
        if (x >= from && x < to)
            ++hist.freq[int((x - from) / hist.bin_size)];
        else if (x == to)
            ++hist.freq.back();
    return hist;
}

Histogram make_histogram(TestResult const& result, int bin_cnt)
{
    return make_histogram(result, result.min_time, result.max_time, bin_cnt);
}

std::ostream& operator<<(std::ostream& out, Histogram const& hist) {
    out << hist.from << ' ' << hist.to << ' ' << hist.bin_cnt << '\n';
    for (int cnt : hist.freq)
        out << cnt << ' ';
    return out << endl;
}

scl_t random_scalar() {
    static default_random_engine gen;
    uniform_real_distribution<scl_t> dist;
    return dist(gen);
}

default_random_engine AlignedData::gen;

AlignedData::AlignedData(size_t size, size_t alignment) {
    size_t space = size*sizeof(scl_t) + alignment - 1;
    void* ptr = orig = malloc(space);
    data = (scl_t*)align(alignment, size, ptr, space);
    uniform_real_distribution<scl_t> dist;
    generate(data, data+size, [&]() {return dist(gen); });
}

//AlignedData::~AlignedData() {
//    free(orig);
//}

vector<scl_t> random_vector(int size) {
    vector<scl_t> vec(size);
    static default_random_engine gen;
    uniform_real_distribution<scl_t> dist;
    generate(vec.begin(), vec.end(), [&]() {return dist(gen); });
    return vec;
}
