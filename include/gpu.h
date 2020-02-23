#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
cublasHandle_t gpu_handle(void);
float* malloc_gpu(size_t n);
void check_error(cudaError_t status);
void random_normal_gpu(int n, float *gpu_x);
void memset_gpu(float *gpu_x, float val, size_t n);

void gpu_push_array(float *x_gpu, float *x, size_t n);
void gpu_pull_array(float *x_gpu, float *x, size_t n);

