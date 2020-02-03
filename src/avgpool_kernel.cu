#include "layers.h"

#ifdef GPU
#include "gpu.h"
#endif


__global__ void avgpool_forward_gpu_kernel(float *output, float *input, int H, int W, int C, int FH, int FW, int FC, int out_h, int out_w, int stride) {

  int b = blockIdx.x;
  int i = threadIdx.x;
  int j = threadIdx.y;

  for(int k = 0; k < FC; k++) {

    int out_idx = b*out_h*out_w*FC + i*out_w*FC + j*FC + k;
    output[out_idx] = 0.0;
 
    for(int n = 0; n < FH; n++) {
      for(int m = 0; m < FW; m++) {
        int im_row = i*stride + n;
        int im_col = j*stride + m;
        int idx = b*H*W*C + im_row*W*C + im_col*C + k;
        output[out_idx] += input[idx];
      }
    }
    output[out_idx] /= (float)(FH*FW);
  }
}



void AvgPool::forward_gpu() {

  int out_w = (W + 2*pad - FW)/stride + 1;
  int out_h = (H + 2*pad - FH)/stride + 1;
  dim3 d = {(unsigned int)out_w, (unsigned int)out_h, 1};
  avgpool_forward_gpu_kernel<<<batch, d>>>(output, input, H, W, C, FH, FW, FC, out_w, out_h, stride);
}

    
__global__ void avgpool_backward_gpu_kernel(float *m_delta, float *delta, int H, int W, int C, int FH, int FW, int FC, int out_h, int out_w, int stride) {

  int b = blockIdx.x;
  int i = threadIdx.x;
  int j = threadIdx.y;

  for(int k = 0; k < FC; k++) {

    int out_idx = b*out_h*out_w*FC + i*out_w*FC + j*FC + k;
    m_delta[out_idx] = 0.0;
    for(int n = 0; n < FH; n++) {
      for(int m = 0; m < FW; m++) {
        int im_row = i*stride + n;
        int im_col = j*stride + m;
        int idx = b*H*W*C + im_row*W*C + im_col*C + k;
        m_delta[idx] += delta[out_idx]/(float)(FH*FW);
      }
    }

  }
}




void AvgPool::backward_gpu(float *delta) {

  int out_w = (W + 2*pad - FW)/stride + 1;
  int out_h = (H + 2*pad - FH)/stride + 1;
  dim3 d = {(unsigned int)out_w, (unsigned int)out_h, 1};

  avgpool_backward_gpu_kernel<<<batch, d>>>(output, input, H, W, C, FH, FW, FC, out_w, out_h, stride);
}
