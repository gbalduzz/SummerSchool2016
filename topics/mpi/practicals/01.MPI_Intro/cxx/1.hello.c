/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS Summer School.        *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS take no responsibility for the use of the enclosed      *
 * teaching material.                                           *
 *                                                              *
 * Purpose: a simple MPI-program printing "hello world!"        *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

/* Write a minimal  MPI program which prints "hello world by each MPI process  */

#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);

  int id;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  std::cout<<"Hello from "<<id<<" of "<<size<<std::endl;
  
  MPI_Finalize();
  return 0;
}
