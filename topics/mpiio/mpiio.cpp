#include <iostream>
#include <mpi.h>
using std::cout;

int main(int argc, char** argv){
  int nsize = 1000000;
  int rank, size;
  MPI_File fh;
  MPI_Status status;


  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double* data = new double[nsize];
  const int offset = rank*sizeof(float)*nsize;
  double t1,t2;
  std::fill_n(data,nsize,rank);

  // Simple view
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  MPI_File_open(MPI_COMM_WORLD, "view.bin", MPI_MODE_WRONLY + MPI_MODE_CREATE,
      MPI_INFO_NULL, &fh);
  MPI_File_set_view(fh, offset, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);

  MPI_File_write(fh, data, nsize, MPI_FLOAT, &status);

  MPI_File_close(&fh);
  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();
  if(rank ==0) cout<<"Time for I/O - VIEW (sec) = "<<t2-t1<<std::endl;

  // 3. COLLECTIVE
  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  MPI_File_open(MPI_COMM_WORLD, "view.bin", MPI_MODE_WRONLY + MPI_MODE_CREATE,
                MPI_INFO_NULL, &fh);
// TODO
  MPI_File_close(&fh);
  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();
  if(rank ==0) cout<<"Time for I/O - COLLECTIVE (sec) = "<<t2-t1<<std::endl;

  delete[] data;
  MPI_Finalize();
}

/*
PROGRAM mpiio

implicit none
include "mpif.h"



!allocate main array
allocate(data(nsize))

!initialize array
do i=1,nsize
   data(i) = real(rank)
enddo

!write data (one file)

! 1 WRITE AT
times=0.0
call MPI_barrier(MPI_COMM_WORLD,ierror)
times(1) = MPI_Wtime()
offset = rank*4*nsize
call MPI_file_open(MPI_COMM_WORLD, 'writeat.bin', MPI_MODE_WRONLY + MPI_MODE_CREATE,&
                   MPI_INFO_NULL, fh, ierror)
call MPI_file_write_at(fh, offset, data, nsize, MPI_REAL, status, ierror)

call MPI_file_close(fh,ierror)
call MPI_barrier(MPI_COMM_WORLD,ierror)
times(2) = MPI_Wtime()
if(rank == 0)write(*,*)"Time for I/O - WRITE AT(sec) = ", times(2) - times(1)

! 2. SIMPLE VIEW

call MPI_barrier(MPI_COMM_WORLD,ierror)
times(1) = MPI_Wtime()
offset = rank*4*nsize
call MPI_file_open(MPI_COMM_WORLD, 'view.bin', MPI_MODE_WRONLY + MPI_MODE_CREATE,&
                   MPI_INFO_NULL, fh, ierror)
call MPI_file_set_view(fh, offset, MPI_REAL, MPI_REAL, 'native', MPI_INFO_NULL, ierror)

call MPI_file_write(fh, data, nsize, MPI_REAL, status, ierror)

call MPI_file_close(fh,ierror)
call MPI_barrier(MPI_COMM_WORLD,ierror)
times(2) = MPI_Wtime()
if(rank == 0)write(*,*)"Time for I/O - VIEW (sec) = ", times(2) - times(1)

! 3. COLLECTIVE

call MPI_barrier(MPI_COMM_WORLD,ierror)
times(1) = MPI_Wtime()
offset = rank*4*nsize
call MPI_file_open(MPI_COMM_WORLD, 'collective.bin', MPI_MODE_WRONLY + MPI_MODE_CREATE,&
                   MPI_INFO_NULL, fh, ierror)
call MPI_file_set_view(fh, offset, MPI_REAL, MPI_REAL, 'native', MPI_INFO_NULL, ierror)

call MPI_file_write_all(fh, data, nsize, MPI_REAL, status, ierror)

call MPI_file_close(fh,ierror)
call MPI_barrier(MPI_COMM_WORLD,ierror)
times(2) = MPI_Wtime()
if(rank == 0)write(*,*)"Time for I/O - COLLECTIVE (sec) = ", times(2) - times(1)


!switch off MPI
call MPI_FINALIZE(ierror)


END PROGRAM mpiio
*/
