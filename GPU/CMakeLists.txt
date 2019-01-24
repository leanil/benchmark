list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp REQUIRED)

add_library( config
    config.cpp
    config.h
)

add_executable( gpu_benchmark
  gpu_benchmark.cpp
  eigen/eigen_test.h
  futhark/futhark_test.hpp
  futhark/futhark_test.c
)

target_include_directories(gpu_benchmark PRIVATE ${EIGEN_ROOT})
target_link_libraries(gpu_benchmark ${OpenCL_LIBRARIES})
    
add_sycl_to_target(
  TARGET gpu_benchmark
  SOURCES gpu_benchmark.cpp
)

add_subdirectory(eigen)
add_subdirectory(futhark)