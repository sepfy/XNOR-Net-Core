#include "layers.h"

#ifdef GPU
#include "gpu.h"
#endif

__global__ void relu_activate_gpu_kernel(float *input, float *output, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;
  output[index] = input[index]*(input[index] >=0);
}

void Relu::relu_activate_gpu() {

  int grid = batch*((N-1)/256 + 1);
    
  relu_activate_gpu_kernel<<<grid, 256>>>(input, output, batch*N);
  cudaDeviceSynchronize();
}


__global__ void relu_backward_gpu_kernel(float *m_delta, float *delta, float *input, float *cut, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;
  m_delta[index] = (cut[index] + delta[index])*(input[index] >= 0);
}



void Relu::relu_backward_gpu(float *delta) {

  int grid = batch*((N-1)/256 + 1);
  relu_backward_gpu_kernel<<<grid, 256>>>(m_delta, delta, input, cut, batch*N);
  cudaDeviceSynchronize();
}


__global__ void leaky_activate_gpu_kernel(float *input, float *output, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;
  output[index] = (input[index] >= 0 ? input[index] : 0.1*input[index]);
}

void Relu::leaky_activate_gpu() {

  int grid = batch*((N-1)/256 + 1);
  leaky_activate_gpu_kernel<<<grid, 256>>>(input, output, batch*N);
  cudaDeviceSynchronize();
}


__global__ void leaky_backward_gpu_kernel(float *m_delta, float *delta, float *input, float *cut, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;
  m_delta[index] = (cut[index] + delta[index])*(input[index] >= 0 ? 1.0 : 0.1);
}



void Relu::leaky_backward_gpu(float *delta) {

  int grid = batch*((N-1)/256 + 1);
  leaky_backward_gpu_kernel<<<grid, 256>>>(m_delta, delta, input, cut, batch*N);
  cudaDeviceSynchronize();
}

