#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <CL/cl.h>

auto to_seconds(cl_ulong t1, cl_ulong t2){ return (std::max(t1, t2) - std::min(t1, t2)) / 1000.0 /1000.0 /1000.0; }

int main()
{
	using T = float;
	const size_t szmin = ((size_t)1) << 2;
    const size_t szmax = ((size_t)1) << 28;
    
    int ndev = 0;
    cudaGetDeviceCount(&ndev);

    for(int k=0; k<ndev; ++k)
    {
        cudaSetDevice(k);

        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, k);
        
        std::cout << "Device: " << prop.name << "\n";
        for(auto i=szmin; i<=szmax; i *= 2)
        {
            T* ptr = new T[i];

            void* device_ptr;
            cudaMalloc(&device_ptr, i*sizeof(T));
            
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaMemcpy(device_ptr, ptr, i*sizeof(T), cudaMemcpyHostToDevice);
            auto t1 = std::chrono::high_resolution_clock::now();
            //cudaMemcpy(hostArray,deviceArray,bytes,cudaMemcpyDeviceToHost);
            cudaFree(device_ptr);
            auto sec = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() / 1000.0 / 1000.0 / 1000.0;
            auto GB = (i*sizeof(T))/1024.0 / 1024.0 / 1024.0;
            std::cout << i*sizeof(T)/1024 << "   dt = " << sec << "   " << GB/sec  <<" GB/s\n";

            delete[] ptr;
        }
    }
	return 0;
}