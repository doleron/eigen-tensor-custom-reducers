cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME mnist_example)
project(${PROJECT_NAME} CXX)

if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release) 
endif() 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(libs/eigen/install.txt)

add_executable(reducer_example ${PROJECT_SOURCE_DIR}/src/reducer_example.cpp)
target_compile_options(reducer_example PRIVATE -Wall -Wextra -pedantic -Wno-enum-compare -Wno-maybe-uninitialized -Wno-uninitialized -Wno-unused-parameter -Wno-array-bounds)
