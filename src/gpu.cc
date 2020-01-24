#include <stdio.h>
#ifdef GPU
#include "gpu.h"

cublasHandle_t gpu_handle() {
  static cublasHandle_t handle;
  if(!handle) {
    cublasCreate(&handle);
  }
  return handle;
}

void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        exit(0);
        //assert(0);
        //snprintf(buffer, 256, "CUDA Error: %s", s);
        //error(buffer);
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        exit(0);
        //assert(0);
        //snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        //error(buffer);
    }
}



float* malloc_gpu(size_t n) {
  float *x_gpu;
  size_t size = n*sizeof(float);
  cudaError_t status = cudaMallocManaged(&x_gpu, size);
  check_error(status);
  return x_gpu;
}
#endif
