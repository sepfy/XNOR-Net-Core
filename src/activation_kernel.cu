#include "layers.h"

__global__ void relu_activate_gpu_kernel(float *input, float *output, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;
  output[index] = input[index]*(input[index] >=0);
}

void Activation::relu_activate_gpu() {

  int grid = batch*((N-1)/256 + 1);
    
  relu_activate_gpu_kernel<<<grid, 256>>>(input, output, batch*N);
  check_error(cudaGetLastError());
}


__global__ void relu_backward_gpu_kernel(float *m_delta, float *delta, float *input, float *cut, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;
  m_delta[index] = (cut[index] + delta[index])*(input[index] >= 0);
}



void Activation::relu_backward_gpu(float *delta) {

  int grid = batch*((N-1)/256 + 1);
  relu_backward_gpu_kernel<<<grid, 256>>>(m_delta, delta, input, cut, batch*N);
  check_error(cudaGetLastError());
}


__global__ void leaky_activate_gpu_kernel(float *input, float *output, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;
  output[index] = (input[index] >= 0 ? input[index] : 0.1*input[index]);
}

void Activation::leaky_activate_gpu() {

  leaky_activate_gpu_kernel<<<default_grid(batch*N), BLOCK>>>(input, output, batch*N);
  check_error(cudaGetLastError());
}


__global__ void leaky_backward_gpu_kernel(float *m_delta, float *delta, float *input, float *cut, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;
  m_delta[index] = (cut[index] + delta[index])*(input[index] >= 0 ? 1.0 : 0.1);
}



void Activation::leaky_backward_gpu(float *delta) {

  leaky_backward_gpu_kernel<<<default_grid(batch*N), BLOCK>>>(m_delta, delta, input, cut, batch*N);
  check_error(cudaGetLastError());
}

__global__ void softmax_kernel(float *input, float *output, int N) {

    int i = threadIdx.x;
    float tmp = 0;
    float max = 0;
    for(int j = 0; j < N; j++)
      if(input[i*N+j] > max)
        max = input[i*N+j];

    for(int j = 0; j < N; j++) {
      output[i*N+j] = exp(input[i*N+j] - max);
      tmp += output[i*N+j];
    }
    for(int j = 0; j < N; j++)
      output[i*N+j] /= tmp;

}

void SoftmaxWithCrossEntropy::forward_gpu() {

  softmax_kernel<<<1, batch>>>(input, output, N);
  check_error(cudaGetLastError());
}

