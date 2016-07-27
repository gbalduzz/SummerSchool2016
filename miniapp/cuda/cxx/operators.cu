//******************************************
// operators
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
//
// implements
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#include "cuda_helpers.h"
#include "data.h"
#include "operators.h"
#include "stats.h"

namespace operators {

// POD type holding information for device
struct DiffusionParams {
    int nx;
    int ny;
    double alpha;
    double dxs;
    double *x_old;
    double *bndN;
    double *bndE;
    double *bndS;
    double *bndW;
};

// TODO : explain what the params variable and setup_params_on_device() do
__device__
DiffusionParams params;

void setup_params_on_device(int nx, int ny, double alpha, double dxs)
{
    auto p = DiffusionParams {
        nx,
        ny,
        alpha,
        dxs,
        data::x_old.device_data(),
        data::bndN.device_data(),
        data::bndE.device_data(),
        data::bndS.device_data(),
        data::bndW.device_data()
    };

    cuda_check_status(
        cudaMemcpyToSymbol(params, &p, sizeof(DiffusionParams))
    );
}

namespace kernels {
    __global__
    void stencil_interior(double* S, const double *U)
    {
      // TODO : implement the interior stencil
      // EXTRA : can you make it use shared memory?
      extern __shared__ double buffer[];
      const int nx = params.nx;      
      const int ny = params.ny;      
      const int gi = (blockDim.x-2)*blockIdx.x+threadIdx.x;
      const int gj = (blockDim.y-2)*blockIdx.y+threadIdx.y;
      if( gi >= nx or gj >= ny) return;      
      const int li = threadIdx.x;
      const int lj = threadIdx.y;
      const int lpos = li+blockDim.x*lj;
      const int gpos = gi+nx*gj;
      if(gi == 0) // north
	buffer[lpos] = params.bndN[gj];
      if(gi == ny-1) // south
	buffer[lpos] = params.bndS[gj];
      if(gj == 0) // west
	buffer[lpos] = params.bndW[gi];
      if(gj == nx-1) // east
	buffer[lpos] = params.bndE[gi];
      else
	buffer[lpos] = U[gpos];
      syncthreads();
      if(li > 0 and li < blockDim.y and
	 lj > 0 and lj < blockDim.x)
	S[gpos] = -(4. + params.alpha) * buffer[lpos] +
	  buffer[lpos-1] + buffer[lpos+1] +
	  buffer[lpos+params.nx] + buffer[lpos-params.nx] +
	  params.alpha * params.x_old[lpos] + 
	  params.dxs * buffer[lpos] * (1. - buffer[lpos]);
    }
} //  kernels

  void diffusion(data::Field const& U, data::Field &S)
{
    using data::options;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::x_old;

    double dxs = 1000. * (options.dx * options.dx);
    double alpha = options.alpha;
    int nx = options.nx;
    int ny = options.ny;

    static bool is_initialized = false;
    if(!is_initialized) {
        setup_params_on_device(nx, ny, alpha, dxs);
        is_initialized = true;
    }

    // TODO: what is the purpose of the following?
    auto calculate_grid_dim = [] (size_t n, size_t block_dim) {
        return (n+block_dim-1)/block_dim;
    };

    // TODO: apply stencil to the interior grid points

    cuda_check_last_kernel("interior point stencil kernel launch");
    constexpr int threads = 32;
    const dim3 threads_grid(threads,threads);
    const dim3 blocks_grid(calculate_grid_dim(nx+2, threads-2),
			   calculate_grid_dim(ny+2, threads-2));
    const int cache_size = (threads)*(threads);
    kernels::stencil_interior
      <<<blocks_grid, threads_grid, cache_size>>>
      (S.device_data(), U.device_data());
    // check
       cuda_check_last_kernel("east-west stencil kernel");

}

} // namespace operators
  
