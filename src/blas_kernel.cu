#include "layers.h"
#include "gpu.h"

/*
__global__ void row_sum_gpu_kernel(float *A, float *B) {

  int j = blockIdx.x;
  int i = threadIdx.x;

  B
  for(int j = 0; j < M; j++)
    for(int i = 0; i < N; i++)
      B[j] += A[i*M+j];
}
*/


void row_sum_gpu(int N, int M, float *A, float *B) {

  for(int i = 0; i < M; i++)
    cublasSasum(gpu_handle(), N, (A + i*M), 1, (B + i));
  //row_sum_gpu_kernel<<<N, M>>>(n, x, grad_x, m_x, v_x, a.beta1, a.beta2, m_lr, a.epsilon);
  cudaDeviceSynchronize();

}


