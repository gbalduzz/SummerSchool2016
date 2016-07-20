/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS Summer School.        *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS, take no responsibility for the use of the enclosed     *
 * teaching material.                                           *
 *                                                              *
 * Purpose: Measuring bandwidth using a ping-pong               *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

/*
 * NOTE: make a reservation with two nodes:
 * salloc ... -N 2 -n 2 ....
 * start mpi using 2 nodes with one process per node:
 * srun -N 1 -n 2 .......
 * use gnuplot to plot the result:
 * gnuplot bandwidth.gp
 *
 * Advanced: try on only one node, explain the bandwidth values
 * srun -N 2 -n 2 .......
 */

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <mpi.h>

#define NMESSAGES 100
#define INI_SIZE 1
#define FACT_SIZE 2
#define REFINE_SIZE_MIN (1*1024)
#define REFINE_SIZE_MAX (16*1024)
#define SUM_SIZE (1*1024)
#define MAX_SIZE (1<<29) /* 512 MBytes */

int main(int argc, char *argv[])
{
    int my_rank, k;
    int length_of_message;
    double start, stop, time, transfer_time;
    MPI_Status status;
    char* buffer;
    std::ofstream f;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    buffer = new char[MAX_SIZE];

    if (my_rank == 0) {
        f.open("bandwidth.dat");
    }

    length_of_message = INI_SIZE;

    while(length_of_message <= MAX_SIZE) {
      /* Write a loop of NMESSAGES iterations which do a ping pong.
       * Make the size of the message variable and display the bandwidth for each of them.
       * What do you observe? (plot it)
       */
      start = MPI_Wtime();
      // **************
      for(int iter=0; iter<NMESSAGES; iter++){
	if( my_rank == 0){
	  MPI_Send(buffer, length_of_message, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
	  MPI_Recv(buffer, length_of_message, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &status);
	}
	if(my_rank == 1){
	  MPI_Recv(buffer, length_of_message, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
	  MPI_Send(buffer, length_of_message, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
	}	  
      }
	// **************
        stop = MPI_Wtime();
        if (my_rank == 0) {
            time = stop - start;

            transfer_time = time / (2 * NMESSAGES);

	    f<<length_of_message<<"\t"
	     <<transfer_time<<"\t"
	     <<(length_of_message / transfer_time)/(1024*1024)
	     <<std::endl;
         
	    //           printf("%s", output_str);

        }
        if (length_of_message >= REFINE_SIZE_MIN && length_of_message < REFINE_SIZE_MAX) {
            length_of_message += SUM_SIZE;
        } else {
            length_of_message *= FACT_SIZE;
        }

    }

    if (my_rank == 0) {
      f.close();
    }
    MPI_Finalize();
    delete[] buffer;
    return 0;
}
