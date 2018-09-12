# CMakeList.txt : CMake project for fusion checks, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("fusion checks")

link_directories(${MKLROOT}/lib/intel64)

# Add source to this project's executable.
add_executable (fusionchecks
                    "fusion checks.cpp"
                    "utility.h"
                    "utility.cpp"
                    "eigen.hpp"
                    "naive.hpp"
                    "vectorized.hpp")

target_include_directories(fusionchecks PRIVATE 
#                           ${MKLROOT}/include
                           ${EIGENROOT})

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /arch:AVX")
message(${CMAKE_CXX_FLAGS})

#target_link_libraries(fusionchecks
#                        mkl_core.lib
#                        mkl_sequential.lib
#                        mkl_intel_ilp64.lib)

# TODO: Add tests and install targets if needed.
