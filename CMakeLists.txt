cmake_minimum_required(VERSION 3.10)

project(CVXNMF)

option(build_python_bindings "Build Python bindings" ON)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)    
    set(CMAKE_BUILD_TYPE "Release")
endif()
## set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address") Causes issues with python bindings.

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
add_subdirectory(src)

if(build_python_bindings)
    add_subdirectory(bindings)
endif()
