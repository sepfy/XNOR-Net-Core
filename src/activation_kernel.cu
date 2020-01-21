#include "layers.h"

#ifdef GPU
#include "gpu.h"
#endif

__global__ void relu_forward_gpu_kernel(float *input, float *output, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index > size)
    return;
  output[index] = input[index]*(input[index] >=0);
}

void Relu::forward_gpu() {

  int grid = batch*((N-1)/256 + 1);
    
  relu_forward_gpu_kernel<<<grid, 256>>>(input, output, batch*N);
  cudaDeviceSynchronize();
}


