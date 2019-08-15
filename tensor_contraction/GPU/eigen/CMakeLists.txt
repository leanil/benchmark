set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library( eigen_baseline
    baseline.cpp
    baseline.hpp
    baseline.h
    util.h
)
target_include_directories(eigen_baseline PRIVATE ${EIGEN_ROOT})
target_link_libraries(eigen_baseline config Threads::Threads)
add_sycl_to_target(
  TARGET eigen_baseline
  SOURCES baseline.cpp
)

add_library( eigen_1d
    1d.cpp
    1d.hpp
    1d.h
    util.h
)
target_include_directories(eigen_1d PRIVATE ${EIGEN_ROOT})
target_link_libraries(eigen_1d config Threads::Threads)
add_sycl_to_target(
  TARGET eigen_1d
  SOURCES 1d.cpp
)

add_library( eigen_2d
    2d.cpp
    2d.hpp
    2d.h
    util.h
)
target_include_directories(eigen_2d PRIVATE ${EIGEN_ROOT})
target_link_libraries(eigen_2d config Threads::Threads)
add_sycl_to_target(
  TARGET eigen_2d
  SOURCES 2d.cpp
)

add_executable( eigen_test
  eigen_test.cpp
  eigen_test.h
)
target_link_libraries(eigen_test config eigen_baseline eigen_1d)# eigen_2d)