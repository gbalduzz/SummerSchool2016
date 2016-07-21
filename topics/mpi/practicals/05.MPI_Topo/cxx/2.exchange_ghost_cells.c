/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS Summer School.        *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS take no responsibility for the use of the enclosed      *
 * teaching material.                                           *
 *                                                              *
 * Purpose: Exchange ghost cell in 2 directions using a topology*
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

/* Use only 16 processes for this exercise
 * Send the ghost cell in two directions: left to right and right to left
 *
 * process decomposition on 4*4 grid
 *
 * |-----------|
 * | 0| 1| 2| 3|
 * |-----------|
 * | 4| 5| 6| 7|
 * |-----------|
 * | 8| 9|10|11|
 * |-----------|
 * |12|13|14|15|
 * |-----------|
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

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <iostream>

#define SUBDOMAIN 10
#define DOMAINSIZE (SUBDOMAIN+2)

int main(int argc, char *argv[])
{
  int rank, size, i, j, rank_top, rank_bottom, rank_left, rank_right;
  double data[DOMAINSIZE*DOMAINSIZE];
  MPI_Request request;
  MPI_Status status;
  MPI_Comm comm_cart;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size!=16) {
    printf("please run this with 16 processors\n");
    MPI_Finalize();
    exit(1);
  }

  // neighbouring ranks with cartesian grid communicator

  // we do not allow the reordering of ranks here
  // an alternative solution would be to allow the reordering and to use the new communicator for the communication
  // then the MPI library has the opportunity to choose the best rank order with respect to performance
  // CREATE a cartesian communicator (4*4) with periodic boundaries and use it to find your neighboring
  // ranks in all dimensions.
  int dims[]={4,4};
  int periods[]={1,1};
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_cart);
  MPI_Comm_rank(comm_cart, &rank);
  MPI_Cart_shift(comm_cart, 0, 1, &rank_left, &rank_right);
  MPI_Cart_shift(comm_cart, 1, 1, &rank_top,  &rank_bottom);
  for (i=0; i<DOMAINSIZE*DOMAINSIZE; i++) {
    data[i]=rank;
  }
  double bndE[SUBDOMAIN];
  double bndW[SUBDOMAIN];
  double bndN[SUBDOMAIN];
  double bndS[SUBDOMAIN];
  double buffE[SUBDOMAIN];
  double buffW[SUBDOMAIN];
  double buffN[SUBDOMAIN];
  double buffS[SUBDOMAIN];
  //  derived datatype, create a datatype for sending the column
  //MPI_Datatype ColumnType;
  //MPI_Type_vector(SUBDOMAIN, 1, DOMAINSIZE, MPI_DOUBLE, &ColumnType);
  //MPI_Type_commit(&ColumnType);
  for(int i=0; i<SUBDOMAIN; i++){
    buffN[i] = data[i+DOMAINSIZE+1];
    buffS[i] = data[i+ DOMAINSIZE*(DOMAINSIZE-1)+1];
    buffE[i] = data[i*DOMAINSIZE+ 2*DOMAINSIZE-2];
    buffW[i] = data[i*DOMAINSIZE+ DOMAINSIZE+1];

    bndW[i] = rank;bndN[i] = rank;bndS[i] = rank;bndE[i] = rank;
  }
  //  ghost cell exchange with the neighbouring cells in all directions
  //  to the top
  MPI_Request rq;
  MPI_Isendrecv(buffN,    SUBDOMAIN, MPI_DOUBLE, rank_top, 0,
               bndN, SUBDOMAIN, MPI_DOUBLE, rank_top, 0, comm_cart, &rq);

  //  to the bottom
  MPI_Sendrecv(buffS,    SUBDOMAIN, MPI_DOUBLE, rank_bottom, 0,
               bndS, SUBDOMAIN, MPI_DOUBLE, rank_bottom, 0, comm_cart, &status);

  MPI_Wait(&rq,&status);
/*
 ///  to the left
  MPI_Sendrecv(buffW,   1, MPI_DOUBLE, rank_left, 0,
               bndW, 1, MPI_DOUBLE, rank_right, 0, comm_cart, &status);
  //  to the right
  MPI_Sendrecv(buffE, 1, MPI_DOUBLE, rank_right, 0,
               bndE, 1, MPI_DOUBLE, rank_left, 0, comm_cart, &status);
*/
  for(int i=0; i<SUBDOMAIN; i++){
    data[i+ 1] = bndN[i];
    data[i+ DOMAINSIZE*(DOMAINSIZE-1)] = bndS[i];
    data[i*DOMAINSIZE+ 2*DOMAINSIZE-1] = bndE[i];
    data[i*DOMAINSIZE +DOMAINSIZE] = bndW[i];
  }

  if (rank==9) {
    printf("data of rank 9 after communication\n");
    std::cout<<"top: "<<rank_top<<" bottom "<<rank_bottom<<std::endl;
    std::cout<<"left: "<<rank_left<<" right "<<rank_right<<std::endl;
    for (j=0; j<DOMAINSIZE; j++) {
      for (i=0; i<DOMAINSIZE; i++) {
        std::cout<< data[i+j*DOMAINSIZE]<<" ";
      }
      std::cout<<std::endl;
    }
  }

 // MPI_Type_free(&ColumnType);
  MPI_Comm_free(&comm_cart);
  MPI_Finalize();

  return 0;
}
