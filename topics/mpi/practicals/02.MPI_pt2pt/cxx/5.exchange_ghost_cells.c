/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS Summer School.        *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS take no responsibility for the use of the enclosed      *
 * teaching material.                                           *
 *                                                              *
 * Purpose: Exchange ghost cell in 2 directions                 *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

/* Use only 16 processes for this exercise
 * Send the ghost cell in two directions: top to bottom and bottom to top
 *
 * process decomposition on 4*4 grid
 *
 *  |-----------|
 *  | 0| 1| 2| 3|
 *  |-----------|
 *  | 4| 5| 6| 7|
 *  |-----------|
 *  | 8| 9|10|11|
 *  |-----------|
 *  |12|13|14|15|
 *  |-----------|
 *
 * Each process works on a 10*10 (SUBDOMAIN) block of data
 * the D corresponds to data, g corresponds to "ghost cells"
 * xggggggggggx
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * gDDDDDDDDDDg
 * xggggggggggx
 */

/* Tasks:
 * A. each rank has to find its top and bottom neighbor
 * B. send them the data they need
 *    - top array goes to top neighbor
 *    - bottom array goes to bottom neighbor
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <mpi.h>

#define SUBDOMAIN 10
#define DOMAINSIZE (SUBDOMAIN+2)

int main(int argc, char *argv[])
{
    int rank, size, i, j, rank_bottom, rank_top;
    double data[DOMAINSIZE*DOMAINSIZE];
    MPI_Request request;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size!=16) {
        printf("please run this with 16 processors\n");
        MPI_Finalize();
        exit(1);
    }

    for (i=0; i<DOMAINSIZE*DOMAINSIZE; i++) {
        data[i]=rank;
    }

<<<<<<< HEAD
    rank_bottom=(rank+4)%16;/* find the rank of the top neighbor */
    rank_top=(rank-4+16)%16;/* find the rank of the bottom neighbor */
    double* bottom_data = &data[SUBDOMAIN*(SUBDOMAIN-1)];
=======
    rank_bottom=(rank-4+16)%16;/* find the rank of the top neighbor */
    rank_top=(rank+4)%16;/* find the rank of the bottom neighbor */
    double top_data[SUBDOMAIN];
    double bottom_data[SUBDOMAIN];
    const int bottom_index = DOMAINSIZE*(DOMAINSIZE-1)+1;
    for(int i=0; i<SUBDOMAIN ; i++){
      top_data[i] = data[1+i];
      bottom_data[i] = data[bottom_index+i];
    }

>>>>>>> e23eff97b70c010cf29f0e691981b17b48d7608d

    //  ghost cell exchange with the neighbouring cells to the bottom and to the top using:
    //  a) MPI_Send, MPI_Irecv, MPI_Wait
    //  b) MPI_Isend, MPI_Recv, MPI_Wait
    //  c) MPI_Sendrecv

    //  to the top

    // a)
<<<<<<< HEAD

    // b)

    // c)
   
    MPI_Sendrecv(&data[0],DOMAINSIZE,MPI_DOUBLE,
		 rank_top,0,
		 bottom_data,DOMAINSIZE,MPI_DOUBLE,
		 rank_top,0,
		 MPI_COMM_WORLD,&status);

    //  to the bottom
    // a)

    // b)

    // c)
    MPI_Sendrecv(bottom_data,DOMAINSIZE,MPI_DOUBLE,
		 rank_bottom,0,
		 &data[0],DOMAINSIZE,MPI_DOUBLE,
		 rank_bottom,0,
		 MPI_COMM_WORLD,&status);

    if (rank==9) {
=======
    MPI_Send(top_data,SUBDOMAIN,MPI_DOUBLE,
	     rank_top,0,MPI_COMM_WORLD);
      
    MPI_Send(bottom_data,SUBDOMAIN,MPI_DOUBLE,		
	     rank_bottom,0,MPI_COMM_WORLD);
    MPI_Irecv(data+1,SUBDOMAIN,MPI_DOUBLE,
	     rank_top,MPI_ANY_TAG,
	     MPI_COMM_WORLD,&request);
    MPI_Recv(data+bottom_index,SUBDOMAIN,MPI_DOUBLE,
	     rank_bottom,MPI_ANY_TAG,
	     MPI_COMM_WORLD,&status);
    MPI_Wait(&request,&status);

    
 if (rank==9) {
>>>>>>> e23eff97b70c010cf29f0e691981b17b48d7608d
        printf("data of rank 9 after communication\n");
        for (j=0; j<DOMAINSIZE; j++) {
            for (i=0; i<DOMAINSIZE; i++) {
                printf("%.1f ", data[i+j*DOMAINSIZE]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
