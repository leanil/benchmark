find_package(OpenCL REQUIRED)

add_executable(futhark_test
  futhark_test.cpp
  futhark_test.hpp
  futhark_test.c
  util.h
  ../config.h
)

target_link_libraries(futhark_test
    ${OpenCL_LIBRARIES})