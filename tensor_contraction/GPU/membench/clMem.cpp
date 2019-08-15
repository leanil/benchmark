#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <CL/cl.h>

auto to_seconds(cl_ulong t1, cl_ulong t2){ return (std::max(t1, t2) - std::min(t1, t2)) / 1000.0 /1000.0 /1000.0; }

int main(int argc, char** argv)
{
	using T = float;
	size_t szmin = ((size_t)1) << 2;
	size_t szmax = ((size_t)1) << 28;

    if (argc > 1)
        szmin = szmax = atoi(argv[1]);

	cl_int status;
	std::vector<cl_platform_id> plats;
	{
		cl_uint n = 0;
		clGetPlatformIDs(0, nullptr, &n);
		plats.resize(n);
		clGetPlatformIDs(n, plats.data(), 0);
	}

	//for(auto p : plats)
    auto p = plats.front();
	{
		std::string pname; 
		{
			size_t sz = 0;
			clGetPlatformInfo(p, CL_PLATFORM_NAME, 0, nullptr, &sz);
			pname.resize(sz);
			clGetPlatformInfo(p, CL_PLATFORM_NAME, sz, (void*)pname.data(), 0);
		}

		cl_uint nd = 0;
		clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd); 

		std::vector<cl_device_id> devs(nd);
		clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), 0);

		//for(auto d : devs)
        auto d = devs.front();
		{
			std::string dname; 
			{
				size_t sz = 0;
				clGetDeviceInfo(d, CL_DEVICE_NAME, 0, nullptr, &sz);
				dname.resize(sz);
				clGetDeviceInfo(d, CL_DEVICE_NAME, sz, (void*)dname.data(), 0);
			}

			std::string fn = pname + dname + ".txt";
			std::cout << fn << "\n";

			cl_context_properties cps[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)p, 0};

			auto ctx = clCreateContext(cps, 1, &d, nullptr, nullptr, &status);
			if(status != CL_SUCCESS){ std::cout << "clCreateContext failed."; return -1; }

			cl_command_queue_properties cqs[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
			auto q = clCreateCommandQueueWithProperties(ctx, d, cqs, &status);
			//auto q = clCreateCommandQueue(ctx, d, CL_QUEUE_PROFILING_ENABLE, &status);
			if(status != CL_SUCCESS){ std::cout << "clCreateCommandQueue failed."; return -1; }

			for(auto i=szmin; i<=szmax; i *= 2)
			{
				T* ptr = new T[i];
				
				auto b1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, i*sizeof(T), nullptr, &status);
				if(status != CL_SUCCESS){ std::cout << "clCreateBuffer 1 failed."; return -1; }
				//auto b2 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS, i*sizeof(T), nullptr, &status);
				//if(status != CL_SUCCESS){ std::cout << "clCreateBuffer 2 failed."; return -1; }
				
				//status = clEnqueueMigrateMemObjects(q, 1, &b2, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, nullptr, nullptr);
				//if(status != CL_SUCCESS){ std::cout << "clEnqueueMigrateMemObjects 2 failed."; return -1; }

				//status = clFinish(q);
				//if(status != CL_SUCCESS){ std::cout << "clFinish 1 failed."; return -1; }

				//cl_event e;
				
				status = clEnqueueMigrateMemObjects(q, 1, &b1, 0, 0, nullptr, nullptr);
				if(status != CL_SUCCESS){ std::cout << "clEnqueueMigrateMemObjects 1 failed."; return -1; }
        status = clFinish(q);
        
				auto t0 = std::chrono::high_resolution_clock::now();
				status = clEnqueueWriteBuffer(q, b1, CL_TRUE, 0, i*sizeof(T), ptr, 0, nullptr, nullptr);
				if(status != CL_SUCCESS){ std::cout << "clEnqueueWriteBuffer 1 failed."; return -1; }

				//status = clFinish(q);
				auto t1 = std::chrono::high_resolution_clock::now();
				if(status != CL_SUCCESS){ std::cout << "clFinish 2 failed."; return -1; }

				auto sec = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() / 1000.0 / 1000.0 / 1000.0;
				auto GB = (i*sizeof(T))/1024.0 / 1024.0 / 1024.0;
				std::cout << i*sizeof(T)/1024 << "   dt = " << sec << "   " << GB/sec  <<" GB/s\n";

				/*cl_ulong times[4];
				status = clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &times[0], nullptr);
				if(status != CL_SUCCESS){ std::cout << "clGetEventProfilingInfo 1 failed."; return -1; }
				status = clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &times[1], nullptr);
				if(status != CL_SUCCESS){ std::cout << "clGetEventProfilingInfo 2 failed."; return -1; }
				status = clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &times[2], nullptr);
				if(status != CL_SUCCESS){ std::cout << "clGetEventProfilingInfo 3 failed."; return -1; }
				status = clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &times[3], nullptr);
				if(status != CL_SUCCESS){ std::cout << "clGetEventProfilingInfo 4 failed."; return -1; }

				std::cout << i*sizeof(T)/1024 << " Kbytes   dt1 = " << to_seconds(times[0], times[1])
					                     << " dt2 = " << to_seconds(times[1], times[2])
					                     << " dt3 = " << to_seconds(times[2], times[3]) << "\n";*/

				//clReleaseEvent(e);
				//clReleaseMemObject(b2);
				clReleaseMemObject(b1);
				delete[] ptr;
			}

			clReleaseCommandQueue(q);
			clReleaseContext(ctx);
		}
	}
	return 0;
}