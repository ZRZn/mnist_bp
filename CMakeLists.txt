cmake_minimum_required(VERSION 3.8)
project(mnist_bp)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp NET.cpp NET.hpp)
add_executable(mnist_bp ${SOURCE_FILES})