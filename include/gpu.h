#include <cublas_v2.h>
#include <cuda_runtime.h>

cublasHandle_t gpu_handle(void);
float* malloc_gpu(size_t n);
