add_executable (mat_sum mat_sum.cpp)
target_compile_features(mat_sum PRIVATE cxx_std_17)
target_include_directories(mat_sum PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(mat_sum PRIVATE util benchmark)
