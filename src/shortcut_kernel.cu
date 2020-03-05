#include "layers.h"


__global__ void shortcut_forward_gpu_kernel(float *input, float *output, float *identity, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;

  output[index] = input[index] + identity[index];
}


void Shortcut::forward_gpu() {

  size_t size = batch*h*w*c;
  shortcut_forward_gpu_kernel<<<default_grid(size), BLOCK>>>(input, output, identity, size);
  check_error(cudaGetLastError());

}

__global__ void shortcut_backward_gpu_kernel(float *delta, float *m_delta, float *cut, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;

  m_delta[index] = delta[index];
  cut[index] = delta[index];
}




void Shortcut::backward_gpu(float *delta) {

  size_t size = batch*h*w*c;
  shortcut_backward_gpu_kernel<<<default_grid(size), BLOCK>>>(delta, m_delta, activation->cut, size);
  check_error(cudaGetLastError());

}

