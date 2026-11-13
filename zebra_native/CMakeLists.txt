cmake_minimum_required(VERSION 3.16)
project(zebra_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# No floating-point reassociation: scores must match LightGBM bit for bit.
add_compile_options(-Wall -Wextra -fno-fast-math)

add_executable(check_trees tools/check_trees.cpp)
