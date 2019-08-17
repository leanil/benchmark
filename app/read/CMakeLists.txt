add_executable (read read.cpp bench.s)
target_compile_features(read PRIVATE cxx_std_17)
target_include_directories(read PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(read PRIVATE util benchmark)
