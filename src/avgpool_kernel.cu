#include "layers.h"

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



void AvgPool::forward_gpu() {

  avgpool_forward_gpu_kernel<<<batch, C>>>(output, input, H, W, C);
  check_error(cudaGetLastError());
}

    
__global__ void avgpool_backward_gpu_kernel(float *m_delta, float *delta, int H, int W, int C) {


  int b = blockIdx.x;
  int k = threadIdx.x;

  int out_idx = b*C + k;

  for(int n = 0; n < H; n++) {
    for(int m = 0; m < W; m++) {
      int idx = b*H*W*C + n*W*C + m*C + k;
      m_delta[idx] = delta[out_idx]/(float)(H*W);
    }
  }

}


void AvgPool::backward_gpu(float *delta) {

  avgpool_backward_gpu_kernel<<<batch, C>>>(m_delta, delta, H, W, C);
  check_error(cudaGetLastError());
}
