#include <cstdio>
#include <iostream>
#include <fstream>

namespace kernels{
__global__
void diffusion(const double x0[], double x1[], const int nx, const int ny, const double dt) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  const int j = threadIdx.y + blockDim.y * blockIdx.y;
  const int width = nx+2;
  const int pos =  i + j*width;

  x1[pos] = x0[pos] + dt * (-4.*x0[pos]
                            + x0[pos-width] + x0[pos+width]
                            + x0[pos-1]  + x0[pos+1]);
}
} // kernels

void diffusion_gpu(const double *x0, double *x1, int nx, int ny, double dt)
{
  int i, j;
  auto width = nx+2;

#ifdef OPENACC_DATA
  // TODO: Offload the following two loops on GPU
#pragma acc parallel loop present(x1,x0) independent  collapse(2)
    for (j = 1; j < ny+1; ++j) {
        for (i = 1; i < nx+1; ++i) {
            auto pos = i + j*width;
            x1[pos] = x0[pos] + dt * (-4.*x0[pos]
                                      + x0[pos-width] + x0[pos+width]
                                      + x0[pos-1]  + x0[pos+1]);
        }
    }
}
#else
  // TODO: Offload the following two loops on GPU, x0 and x1 are allocated by
  // CUDA
  const int thread = 32;
  const dim3 grid(gridSize(nx,thread), gridSize(ny,thread))
  kernel::diffusion<<<gridSize(nx*ny,thread), dim3(thread,thread)>>>(x0, x1, nx, ny, dt);
#endif

template<typename T>
void copy_gpu(T *dst, const T *src, int n)
{
  int i;

#ifdef OPENACC_DATA
  // TODO: Offload the copying on GPU
#pragma omp parallel loop independent present(dst,src)
    for (i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
#else
  // TODO: Offload the copying on GPU, dst and src are allocated by CUDA
  cudaMemcpy(dst, src, n*sizeof(T), cudaMemcpyDeviceToDevice);
#endif
}


template <typename T>
void fill_gpu(T *v, T value, int n)
{
  int i;
#ifdef OPENACC_DATA
  // TODO: Fill data in GPU
  #pragma omp parallel loop independent present(v)
    for (i = 0; i < n; ++i)
        v[i] = value;
#else
  // TODO: Fill data in GPU, v is allocated by CUDA
  cudaMemset(v, value, n**sizeof(T));
#endif

}

void write_to_file(int nx, int ny, double* data) {
  {
    FILE* output = fopen("output.bin", "w");
    fwrite(data, sizeof(double), nx * ny, output);
    fclose(output);
  }

  std::ofstream fid("output.bov");
  fid << "TIME: 0.0" << std::endl;
  fid << "DATA_FILE: output.bin" << std::endl;
  fid << "DATA_SIZE: " << nx << ", " << ny << ", 1" << std::endl;;
  fid << "DATA_FORMAT: DOUBLE" << std::endl;
  fid << "VARIABLE: phi" << std::endl;
  fid << "DATA_ENDIAN: LITTLE" << std::endl;
  fid << "CENTERING: nodal" << std::endl;
  fid << "BRICK_SIZE: 1.0 1.0 1.0" << std::endl;
}
