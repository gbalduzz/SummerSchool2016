/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS Summer School.        *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS take no responsibility for the use of the enclosed      *
 * teaching material.                                           *
 *                                                              *
 * Purpose: Parallel sum using a ping-pong                      *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <iostream>
#include <vector>
#include <mpi.h>


int main (int argc, char *argv[])
{
    int my_rank, size;
    int snd_buf, rcv_buf;
    int right, left;
    int sum(0), i;

    MPI_Status  status;
    MPI_Request request;


    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);


    right = (my_rank + 1) % size; /* get rank of neighbor to your right */
    left  = (my_rank - 1 + size) % size; /* get rank of neighbor to your left */

    std::vector<int> ranks(size);
    /* Implement ring addition code 
     * do not use if (rank == 0) .. else ..
     * every rank sends initialy its rank number to a neighbor, and then sends what
     * it receives from that neighbor, this is done n times with n = number of processes
     * all ranks will obtain the sum.
     */
    ranks[0] = my_rank;
    for(int i=1; i<size; i++){
      MPI_Sendrecv(&ranks[i-1], 1, MPI_INT,
                right, 0,
                &ranks[i], 1, MPI_INT,
                left, 0,
		   MPI_COMM_WORLD, &status);

    }
    for(int i=0; i<ranks.size(); i++) sum += ranks[i];
    std::cout<<"Process "<<my_rank<< "\tSum "<<sum<<"\n";

    MPI_Finalize();
    return 0;
}
