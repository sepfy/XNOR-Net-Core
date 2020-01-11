#ifdef GPU
#include "gpu.h"

cublasHandle_t gpu_handle() {
  static cublasHandle_t handle;
  if(!handle) {
    cublasCreate(&handle);
  }
  return handle;
}

float* malloc_gpu(size_t n) {
  float *x_gpu;
  size_t size = n*sizeof(float);
  cudaMallocManaged(&x_gpu, size);
  return x_gpu;
}
#endif
