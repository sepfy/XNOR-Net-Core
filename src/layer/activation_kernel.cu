#include "layer/activation.h"

void Activation::Init() {

  output = malloc_gpu(batch*N);
  delta_ = malloc_gpu(batch*N);
  cut = malloc_gpu(batch*N);
}

__global__ void relu_activate_gpu_kernel(float *input, float *output, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;
  output[index] = input[index]*(input[index] >=0);
}

void Activation::relu_activate() {

  int grid = batch*((N-1)/256 + 1);
    
  relu_activate_gpu_kernel<<<grid, 256>>>(input, output, batch*N);
  check_error(cudaGetLastError());
}


__global__ void relu_backward_gpu_kernel(float *delta_, float *delta, float *input, float *cut, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;
  delta_[index] = (cut[index] + delta[index])*(input[index] >= 0);
}



void Activation::relu_backward(float *delta) {

  int grid = batch*((N-1)/256 + 1);
  relu_backward_gpu_kernel<<<grid, 256>>>(delta_, delta, input, cut, batch*N);
  check_error(cudaGetLastError());
}


__global__ void leaky_activate_gpu_kernel(float *input, float *output, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;
  output[index] = (input[index] >= 0 ? input[index] : 0.1*input[index]);
}

void Activation::leaky_activate() {

  leaky_activate_gpu_kernel<<<default_grid(batch*N), BLOCK>>>(input, output, batch*N);
  check_error(cudaGetLastError());
}


__global__ void leaky_backward_gpu_kernel(float *delta_, float *delta, float *input, float *cut, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;
  delta_[index] = (cut[index] + delta[index])*(input[index] >= 0 ? 1.0 : 0.1);
}



void Activation::leaky_backward(float *delta) {

  leaky_backward_gpu_kernel<<<default_grid(batch*N), BLOCK>>>(delta_, delta, input, cut, batch*N);
  check_error(cudaGetLastError());
}


void Activation::Forward() {

   switch(activation_type_) {
    case RELU:
      relu_activate();
    case LEAKY:
      leaky_activate();
  }
}

void Activation::Backward(float *delta) {

  switch(activation_type_) {
    case RELU:
      relu_backward(delta);
    case LEAKY:
      leaky_backward(delta);
  }
}

void Activation::LoadParams(std::fstream *rfile, int batch){}
