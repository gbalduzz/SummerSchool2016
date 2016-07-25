cmake_minimum_required(VERSION 3.1)
project(miniapp_mpi)

set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -Wall -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O4 -fstrict-aliasing -ffast-math -mtune=native -march=native -funroll-loops ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=3 -std=c++14  -Wstrict-aliasing=2 -fopenmp")

add_subdirectory(/home/giovanni/Programs/SummerSchool2016/miniapp/mpi/cxx/)

