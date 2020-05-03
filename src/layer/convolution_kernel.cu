#include "layer/convolution.h"
#include "gemm.h"
#include "utils.h"
#include "blas.h"

__global__ void binarize_input_gpu_kernel(float *input, int size) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i >= size) return;
  input[i] > 0 ? input[i] = 1 : input[i] = -1;
    
}

__global__ void input_mean_gpu_kernel(float *avg_filter, float *input, int n, int c) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i >= n) return;

  avg_filter[i] = 0.0;
  for(int k = 0; k < c; k++)
     avg_filter[i] += fabs(input[i*c + k]);
  avg_filter[i] /= (float)c;

}

void Convolution::binarize_input_gpu() {

  int size = batch*H*W; 
  input_mean_gpu_kernel<<<default_grid(size), BLOCK>>>(avg_filter, input, size, C);
  check_error(cudaGetLastError());

  size = batch*out_h*out_w*out_channel;
  binarize_input_gpu_kernel<<<default_grid(size), BLOCK>>>(shared, size);
  check_error(cudaGetLastError());

}

__global__ void binarize_weight_gpu_kernel(float *binary_weight, float *weight, int size) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i >= size) return;
  binary_weight[i] = (weight[i] > 0) ? 1.0 : -1.0;

}

__global__ void weight_mean_gpu_kernel(float *mean, float *weight, int n, int c) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i >= n) return;
  mean[i] = 0.0;
  for(int j = 0; j < c; j++)
    mean[i] += fabs(weight[i*c+j]);
  mean[i] /= (float)c;
}

void Convolution::binarize_weight_gpu() {

  weight_mean_gpu_kernel<<<default_grid(FC), BLOCK>>>
    (mean, weight, FC, out_channel);
  check_error(cudaGetLastError());

  int size = out_channel*FC;
  binarize_weight_gpu_kernel<<<default_grid(size), BLOCK>>>
    (binary_weight, weight, size);
  check_error(cudaGetLastError());

}


__global__ void multi_mean_gpu_kernel(float *output, float *mean, float *k_output, int n, int FC) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i >= n) return;

  int j = i/FC;
  int k = i%FC;
  output[i] *= (mean[k]*k_output[j]);

}


void Convolution::forward_xnor_gpu() {

  binarize_weight_gpu();
  swap_weight();

  for(int i = 0; i < batch; i++)
    im2col_gpu(W, H, C, FW, FH, FC, stride, pad,
      input + i*im_size, shared+i*col_size);

  binarize_input_gpu();
  

  gemm_gpu(TRS_N, TRS_N, batch*out_h*out_w, FC, out_channel, 1, shared, weight, output);


  // Do K = A (*) k
  for(int i = 0; i < batch; i++)
    im2col_gpu(W, H, 1, FW, FH, 1, stride, pad,
      avg_filter + i*W*H, avg_col + i*out_w*out_h*FW*FH);
  gemm_gpu(TRS_N, TRS_N, batch*out_h*out_w, 1, FW*FH, 1.0, avg_col, k_filter, k_output);

  int size = batch*out_w*out_h*FC;
  multi_mean_gpu_kernel<<<default_grid(size), BLOCK>>>(output, mean, k_output, size, FC);
  swap_weight();
}

void Convolution::forward_full_gpu() {

  for(int i = 0; i < batch; i++)
    im2col_gpu(W, H, C, FW, FH, FC, stride, pad,
      input + i*im_size, shared+i*col_size);

  gemm_gpu(TRS_N, TRS_N, batch*out_h*out_w, FC, out_channel, 1, shared, weight, output);

}

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

void Convolution::bias_add_gpu() {

  size_t size = out_w*out_h*batch*FC;
  bias_add_kernel<<<default_grid(size), BLOCK>>>(output, bias, out_w, out_h, FC, size);
  check_error(cudaGetLastError());
}

void Convolution::forward() {
  
  if(xnor) forward_xnor_gpu();
  else forward_full_gpu();
  bias_add_gpu();
}

void Convolution::backward(float *delta) {

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

