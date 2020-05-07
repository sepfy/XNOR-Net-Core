#include "layer/avgpool.h"


void Avgpool::Init() {

  output = malloc_gpu(batch*C);
  delta_ = malloc_gpu(batch*H*W*C);
}

__global__ void avgpool_forward_gpu_kernel(float *output, float *input, int H, int W, int C) {

  int b = blockIdx.x;
  int k = threadIdx.x;

  int out_idx = b*C + k;
  output[out_idx] = 0.0;
 
  for(int n = 0; n < H; n++) {
    for(int m = 0; m < W; m++) {
      int idx = b*H*W*C + n*W*C + m*C + k;
      output[out_idx] += input[idx];
    }
  }
  output[out_idx] /= (float)(H*W);
}



void Avgpool::Forward() {

  avgpool_forward_gpu_kernel<<<batch, C>>>(output, input, H, W, C);
  check_error(cudaGetLastError());
}

    
__global__ void avgpool_backward_gpu_kernel(float *delta_, float *delta, int H, int W, int C) {


  int b = blockIdx.x;
  int k = threadIdx.x;

  int out_idx = b*C + k;

  for(int n = 0; n < H; n++) {
    for(int m = 0; m < W; m++) {
      int idx = b*H*W*C + n*W*C + m*C + k;
      delta_[idx] = delta[out_idx]/(float)(H*W);
    }
  }

}


void Avgpool::Backward(float *delta) {

  avgpool_backward_gpu_kernel<<<batch, C>>>(delta_, delta, H, W, C);
  check_error(cudaGetLastError());
}
