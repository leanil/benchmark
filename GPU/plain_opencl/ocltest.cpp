#include "CL/cl.hpp"
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

void my_assert(int code, string const& message = "") {
    if (code) {
        cerr << "error(" << code << ") " << message << endl;
        exit(1);
    }
}

std::string kernel_code =
"   kernel void sum_kernel(global const float* v, const int n, global float* tmp, local float* ans) {"
"       int id = get_local_id(0), Id = get_group_id(0), step = get_global_size(0);"
"       for(int i=get_global_id(0); i<n; i+=step)"
"           ans[id] += v[i];"
"       barrier(CLK_LOCAL_MEM_FENCE);"
"       int i = 1, m = get_local_size(0);"
"       for(;2*i<m;i*=2);"
"       for(;i>=1;i/=2)"
"           if(id + i < m)"
"               ans[id] += ans[id+i];"
"       if(id == 0)"
"           tmp[Id] = ans[0];"
"   }";

const int n = 1 << 20;
const int group_cnt = 64, worker_cnt = 32;

int main() {
    vector<cl::Platform> platforms;
    my_assert(cl::Platform::get(&platforms));
    vector<cl::Device> devices;
    my_assert(platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices));
    cl::Device device = devices[0];
    cl::Context context({ device });
    cl_int ret;
    cl::Program program(context, kernel_code, true, &ret);
    my_assert(ret, program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
    vector<float> data(n);
    iota(data.begin(), data.end(), 0);
    cl::Buffer buffer(context, CL_MEM_READ_ONLY, sizeof(float) * n);
    cl::Buffer tmp(context, CL_MEM_READ_WRITE, sizeof(float)*group_cnt);
    my_assert(queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(float)*n, data.data()));
    auto test1 = cl::make_kernel<cl::Buffer&, int, cl::Buffer&, cl::LocalSpaceArg>(program, "sum_kernel", &ret);
    my_assert(ret);
    /*auto test2 = cl::make_kernel<cl::Buffer&, cl::Buffer&>(program, "sum_kernel2", &ret);
    my_assert(ret);*/
    auto event1 = test1(cl::EnqueueArgs(queue, cl::NDRange(group_cnt*worker_cnt), cl::NDRange(worker_cnt)), buffer, n, tmp, cl::Local(worker_cnt));
    //auto event2 = test(cl::EnqueueArgs(queue, cl::NDRange(group_cnt)), tmp, n, out);
    vector<float> tmp_data(group_cnt);
    my_assert(queue.enqueueReadBuffer(tmp, CL_TRUE, 0, sizeof(float)*group_cnt, tmp_data.data()));
    queue.finish();
    float ans = 0;
    for (float x : tmp_data)
        ans += x;
    cout << ans << endl;
    ans = 0;
    for (float x : data)
        ans += x;
    cout << ans << endl;
    cl_ulong start, end;
    my_assert(event1.getProfilingInfo(CL_PROFILING_COMMAND_START, &start));
    my_assert(event1.getProfilingInfo(CL_PROFILING_COMMAND_END, &end));
    float time1 = (end - start) * 1.0e-6f;
    //my_assert(event2.getProfilingInfo(CL_PROFILING_COMMAND_START, &start));
    //my_assert(event2.getProfilingInfo(CL_PROFILING_COMMAND_END, &end));
    //float time2 = (end - start) * 1.0e-6f;
    //cout << time1 << " + " << time2 << " = " << time1 + time2 << endl;
    cout << time1 << endl;
}