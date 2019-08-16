add_library(util util.cpp)
target_include_directories(util PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(util PRIVATE benchmark)
