add_executable (array_sum array_sum.cpp bench.s)
target_compile_features(array_sum PRIVATE cxx_std_17)
target_include_directories(array_sum PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(array_sum PRIVATE util benchmark)
