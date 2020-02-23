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



void row_sum_gpu(int N, int M, float *A, float *B) {

  for(int i = 0; i < M; i++)
    cublasSasum(gpu_handle(), N, (A + i*M), 1, (B + i));
  //row_sum_gpu_kernel<<<N, M>>>(n, x, grad_x, m_x, v_x, a.beta1, a.beta2, m_lr, a.epsilon);
  cudaDeviceSynchronize();

}
*/

__global__ void row_sum_gpu_kernel(float *A, float *B, int N, int M) {
    
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index > M)
    return;
  int i = 0;
  B[index] = 0.0;
  for(i = 0; i < M; i++)
    B[index] += A[index*M + i];
}


void row_sum_gpu(int N, int M, float *A, float *B) {

    size_t BLOCK = 512;
    int grid = (M-1)/BLOCK + 1;
    row_sum_gpu_kernel<<<grid, BLOCK>>>(A, B, N, M);
    check_error(cudaGetLastError());
}


__global__ void col_sum_gpu_kernel(float *A, float *B, int N, int M) {
   
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index > N)
    return;
  int i = 0;
  B[index] = 0.0;
  for(i = 0; i < N; i++)
    B[index] += A[i*M + index];
/* 
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index > M)
    return;
  int i = 0;
  for(i = 0; i < M; i++)
    B[index] += A[index*M + i];
*/
}


void col_sum_gpu(int N, int M, float *A, float *B) {

/*
  size_t BLOCK = 512;
  int grid = (N-1)/BLOCK + 1;
  col_sum_gpu_kernel<<<grid, BLOCK>>>(A, B, N, M);
  check_error(cudaGetLastError());
*/
  cudaMemset(B, 0, sizeof(float)*M);
  check_error(cudaGetLastError());
  float alpha = 1.0;
  float beta = 0.0;
  float *e = malloc_gpu(N);
  cudaMemset(e, 1.0, sizeof(float)*N);
  check_error(cudaGetLastError());
  cublasSgemv(gpu_handle(), CUBLAS_OP_T, M, N, &alpha, A, M, e, 1, &beta, B, 1);
  check_error(cudaGetLastError());
  cudaFree(e);
}

/*
void col_sum_gpu(int N, int M, float *A, float *B) {

  memset(B, 0, N*sizeof(float));
  float alpha = 1.0;
  float beta = 0.0;
  float *e = malloc_gpu(N);
  for(int i = 0; i < N; i++)
    e[i] = 1.0;
  cublasSgemv(gpu_handle(), CUBLAS_OP_T, M, N, &alpha, A, M, e, 1, &beta, B, 1);
  cudaDeviceSynchronize();
}
*/

__global__ void bias_add_kernel1(float *output, float *bias,
                         int batch, int size, int channel) {

    int i = threadIdx.x;
    int b = blockIdx.x;
    for(int j = 0; j < channel; j++)
      output[b*size*channel+i*channel+j] += bias[j];

}

// TODO: Integrate the bias add of convolution and connected.
void bias_add_gpu(float *output, float *bias, int batch, int size, int c) {

  bias_add_kernel1<<<batch, size>>>(output, bias, batch, size, c);
}


