#include "layers.h"

#if 1
__global__ void bias_add_kernel(float *output, float *bias,
                         int out_h, int out_w, int FC, int size) {


    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index > size) return;

    int b = index/(out_h*out_w*FC);
    int i = index/(out_w*FC)%out_h ;
    int j = index/FC%out_w;
    int k = index%FC;

    output[b*out_w*out_h*FC + i*out_w*FC + j*FC +k] += bias[k];

}

#endif

__global__ binarize_input_gpu_kernel(float *input, int size) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i >= size) return;
  input[i] > 0 ? input[i] = 1 : input[i] = -1;
    
}

void binarize_input_gpu() {
/*
  for(int b = 0; b < batch; b++) {
    for(int i = 0; i < H; i++) {
      for(int j = 0; j < W; j++) {
        int avg_idx = b*H*W + i*W + j;
        int in_idx = b*im_size + i*W*C + j*C;
        avg_filter[avg_idx] = 0.0;
        for(int k = 0; k < C; k++)
          avg_filter[avg_idx] += fabs(input[in_idx + k]);
        avg_filter[avg_idx] /= (float)C;
      }
    }
  }
*/
  int size = batch*im_size;
  binarize_input_gpu_kernel<<<default_grid(size), BLOCK>>>(input, size);

}
__global__ void binarize_weight_gpu_kernel(float *binary_weight, float *weight, int size) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i >= size) return;
  binary_weight[i] = (weight[i] > 0) ? 1.0 : -1.0;

}

__global__ weight_mean_gpu_kernel(float *mean, float *weight, int n, int c) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i >= n) return;
  mean[i] = 0.0;
  for(int j = 0; j < c; j++)
    mean[i] += fabs(weight[i*c+j]);
  mean[i] /= (float)c;
}

void binarize_weight_gpu() {

  weight_mean_gpu_kernel<<<default_grid(FC), BLOCK>>>(mean, weight, FC, out_channel);
  int size = out_channel*FC;
  binarize_weight_gpu_kernel<<<default_grid(size), BLOCK>>>(binary_weight, weight, size);

}


void Convolution::forward_gpu() {

  for(int i = 0; i < batch; i++)
    im2col_gpu(W, H, C, FW, FH, FC, stride, pad,
      input + i*im_size, shared+i*col_size);

  gemm_gpu(TRS_N, TRS_N, batch*out_h*out_w, FC, out_channel, 1, shared, weight, output);

}

void Convolution::bias_add_gpu() {

  size_t size = out_w*out_h*batch*FC;
  bias_add_kernel<<<default_grid(size), BLOCK>>>(output, bias, out_w, out_h, FC, size);
  check_error(cudaGetLastError());
}

void Convolution::backward_gpu(float *delta) {

  for(int i = 0; i < batch; i++)
    im2col_gpu(W, H, C, FW, FH, FC, stride, pad,
      input + i*im_size, shared+i*col_size);

  gemm_gpu(TRS_T, TRS_N,
           out_channel, FC, out_h*out_w*batch, 1.0,
           shared, delta, grad_weight);
  row_sum_gpu(batch*out_w*out_h, FC, delta, grad_bias);

  gemm_gpu(TRS_N, TRS_T,
       batch*out_w*out_h, out_channel, FC, 1.0,
       delta, weight, shared);

  for(int i = 0; i < batch; i++) {
    col2im_gpu(W, H, C, FW, FH, FC, stride, pad,
      m_delta + i*im_size, shared  + i*col_size);
  }
}



#if 0
__global__ void bias_add_kernel(float *output, float *bias,
                         int batch, int im_size, int channel) {

    int i = threadIdx.x;
    int b = blockIdx.x;
    for(int j = 0; j < channel; j++)
      output[b*im_size*channel+i*channel+j] += bias[j];

}

void Convolution::bias_add_gpu() {

  int size = out_w*out_h;
  bias_add_kernel<<<batch, size>>>(output, bias, batch, size, FC);
  check_error(cudaGetLastError());
}

#endif


