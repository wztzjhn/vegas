cmake_minimum_required(VERSION 4.0)
project(vegas)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(vegas test.cc)
