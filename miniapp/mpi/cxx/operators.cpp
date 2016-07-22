// TODO : halo exchange

//******************************************
// operators.cpp
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#include <iostream>

#include <mpi.h>

#include "data.h"
#include "operators.h"
#include "stats.h"

namespace operators {

void diffusion(const data::Field &U, data::Field &S)
{
  using data::options;
  using data::domain;

  using data::bndE;
  using data::bndW;
  using data::bndN;
  using data::bndS;

    using data::buffE;
  using data::buffW;
  using data::buffN;
  using data::buffS;

  using data::x_old;

    double dxs = 1000. * (options.dx * options.dx);
  double alpha = options.alpha;
  int nx = domain.nx;
  int ny = domain.ny;
  int iend  = nx - 1;
  int jend  = ny - 1;

  MPI_Request request[8];
  MPI_Status status[8];
  int n_request=0;

  if(domain.neighbour_north>=0) {
    for(int j=0; j<nx;j++) buffN[j] = U(0,j);
    MPI_Isend(buffN.data(), nx, MPI_DOUBLE, domain.neighbour_north, 0, domain.comm_cart, request+n_request);
    n_request++;
    MPI_Irecv(bndN.data(),  nx, MPI_DOUBLE, domain.neighbour_north, 0, domain.comm_cart, request+n_request);
    n_request++;
  }
  if(domain.neighbour_south>=0) {
    for(int j=0; j<nx;j++) buffS[j] = U(ny-1,j);
    MPI_Isend(buffS.data(), nx, MPI_DOUBLE, domain.neighbour_south, 0, domain.comm_cart, request+n_request);
    n_request++;
    MPI_Irecv(bndS.data(),  nx, MPI_DOUBLE, domain.neighbour_south, 0, domain.comm_cart, request+n_request);
    n_request++;
  }
  if(domain.neighbour_west>=0) {
    for(int i=0; i<nx;i++) buffW[i] = U(i,ny-1);
    MPI_Isend(buffW.data(), ny, MPI_DOUBLE, domain.neighbour_west, 0, domain.comm_cart, request+n_request);
    n_request++;
    MPI_Irecv(bndW.data(),  ny, MPI_DOUBLE, domain.neighbour_west, 0, domain.comm_cart, request+n_request);
    n_request++;
  }
  if(domain.neighbour_east>=0) {
    for(int i=0; i<nx;i++) buffW[i] = U(i,0);
    MPI_Isend(buffE.data(), ny, MPI_DOUBLE, domain.neighbour_east, 0, domain.comm_cart, request+n_request);
    n_request++;
    MPI_Irecv(bndE.data(),  ny, MPI_DOUBLE, domain.neighbour_east, 0, domain.comm_cart, request+n_request);
    n_request++;
  }

    // the interior grid points
    #pragma omp parallel for
    for (int j=1; j < jend; j++) {
        for (int i=1; i < iend; i++) {
            S(i,j) = -(4. + alpha) * U(i,j)               // central point
                                    + U(i-1,j) + U(i+1,j) // east and west
                                    + U(i,j-1) + U(i,j+1) // north and south
                                    + alpha * x_old(i,j)
                                    + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

  MPI_Waitall(n_request, request, status);
    // the east boundary
    {
        int i = nx - 1;
        for (int j = 1; j < jend; j++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j-1) + U(i,j+1)
                        + alpha*x_old(i,j) + bndE[j]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the west boundary
    {
        int i = 0;
        for (int j = 1; j < jend; j++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j-1) + U(i,j+1)
                        + alpha * x_old(i,j) + bndW[j]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the north boundary (plus NE and NW corners)
    {
        int j = ny - 1;

        {
            int i = 0; // NW corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j-1)
                        + alpha * x_old(i,j) + bndW[j] + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        // north boundary
        for (int i = 1; i < iend; i++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i+1,j) + U(i,j-1)
                        + alpha*x_old(i,j) + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        {
            int i = nx-1; // NE corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j-1)
                        + alpha * x_old(i,j) + bndE[j] + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the south boundary
    {
        int j = 0;

        {
            int i = 0; // SW corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndW[j] + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        // south boundary
        for (int i = 1; i < iend; i++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i+1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        {
            int i = nx - 1; // SE corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndE[j] + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // Accumulate the flop counts
    // 8 ops total per point
    stats::flops_diff +=
        + 12 * (nx - 2) * (ny - 2) // interior points
        + 11 * (nx - 2  +  ny - 2) // NESW boundary points
        + 11 * 4;                                  // corner points
}

} // namespace operators
