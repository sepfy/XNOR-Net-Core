#include <stdio.h>
#ifdef GPU
#include "gpu.h"
#include "utils.h"

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
    //cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error (%d): %s\n", status, s);
        exit(0);
        //assert(0);
        //snprintf(buffer, 256, "CUDA Error: %s", s);
        //error(buffer);
    }

    /*
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
    */
}



float* malloc_gpu(size_t n) {
  float *x_gpu;
  size_t size = n*sizeof(float);
  //cudaError_t status = cudaMallocManaged(&x_gpu, size);
  cudaError_t status = cudaMalloc((void**)&x_gpu, size);
  check_error(status);
  status = cudaMemset(x_gpu, 0, size);
  check_error(status);
  return x_gpu;
}

void random_normal_gpu(int n, float *gpu_x) {
/*
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  curandGenerateUniform(gen, gpu_x, n);
*/
  float *tmp = new float[n];
  random_normal(n, tmp);
  gpu_push_array(gpu_x, tmp, n);
  delete []tmp;
}

void memset_gpu(float *gpu_x, float val, size_t n) {

  float *x = new float[n];
  for(int i = 0; i < n; i++)
    x[i] = val;
  gpu_push_array(gpu_x, x, n);
  check_error(cudaGetLastError());
  delete []x;
}



void gpu_push_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    check_error(status);
}

void gpu_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    check_error(cudaGetLastError());
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

int default_grid(int N) {
  int GRID = (N-1)/BLOCK + 1;
  return GRID;
}

#endif
