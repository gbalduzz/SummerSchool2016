/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS Summer School.        *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS take no responsibility for the use of the enclosed      *
 * teaching material.                                           *
 *                                                              *
 * Purpose: A ping-pong                                         *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

#include <iostream>
#include <mpi.h>

#define PING  0 //message tag
#define PONG  1 //message tag
#define SIZE  1024

int main(int argc, char *argv[])
{
  int my_rank,size;
    float buffer[SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    if(size != 2){
      std::cout<<"Run with 2 ranks \n";
      return -1;
    }
    
    if( my_rank ==0){
      for(int i=0; i<SIZE; i++) buffer[i] = i;
    	MPI_Send(&buffer, SIZE, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
	MPI_Recv(&buffer, SIZE, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &status);
    }

    if(my_rank == 1){
    	MPI_Recv(&buffer, SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
	for(int i=0; i<SIZE; i++) buffer[i] *= 2;
	MPI_Send(&buffer, SIZE, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
    
    if(my_rank == 0){
      for(int i=0; i<5; i++) std::cout<<buffer[i]<<std::endl;
    }
    /* Process 0 sends a message (ping) to process 1.
     * After receiving the message, process 1 sends a message (pong) to process 0.
     */

    MPI_Finalize();
    return 0;
}
