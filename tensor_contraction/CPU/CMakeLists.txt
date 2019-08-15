cmake_minimum_required (VERSION 3.8)

project ("benchmark")

set(CMAKE_CXX_STANDARD 14)

#link_directories(${MKLROOT}/lib/intel64)

add_executable (fusionchecks
                    "fusion_checks.cpp"
                    "utility.h"
                    "utility.cpp"
                    "eigen.hpp"
                    "naive.hpp"
                    "vectorized.hpp"
                    "futhark.hpp"
                    "futhark_lib.h"
                    "futhark_lib.cpp")

target_include_directories(fusionchecks PRIVATE 
#                           ${MKLROOT}/include
                           ${EIGEN_ROOT})

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -msse4.2")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
     SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /arch:SSE2")
endif()

#target_link_libraries(fusionchecks
#                        mkl_core.lib
#                        mkl_sequential.lib
#                        mkl_intel_ilp64.lib)

add_subdirectory(GPU)

