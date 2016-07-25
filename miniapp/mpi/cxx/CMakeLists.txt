find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(main main.cpp data.cpp operators.cpp linalg.cpp stats.cpp)
target_link_libraries(main ${MPI_C_LIBRARIES})
