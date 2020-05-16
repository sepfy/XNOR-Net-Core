#include "layer/binary_conv.h"
#include "gemm.h"
#include "utils.h"
#include "blas.h"


void BinaryConv::Init() {

  output = malloc_gpu(batch*out_w*out_h*filter_channel);
  weight = malloc_gpu(filter_col*filter_channel);
  grad_weight = malloc_gpu(filter_col*filter_channel);
  bias = malloc_gpu(filter_channel);
  grad_bias = malloc_gpu(filter_channel);
  delta_ = malloc_gpu(batch*width*width*channel);

  // Adam optimizer
  m_weight = malloc_gpu(filter_col*filter_channel);
  v_weight = malloc_gpu(filter_col*filter_channel);
  m_bias = malloc_gpu(filter_channel);
  v_bias = malloc_gpu(filter_channel);

  binary_weight = malloc_gpu(filter_col*filter_channel);
  avg_filter = malloc_gpu(batch*im_size);
  avg_col = malloc_gpu(out_w*out_h*filter_width*filter_channel*batch);
  k_filter = malloc_gpu(filter_width*filter_height);
  k_output = malloc_gpu(out_w*out_h*batch);

  memset_gpu(k_filter, 1.0/(float)(filter_width*filter_height), filter_width*filter_height);
  mean = malloc_gpu(filter_channel);

  random_normal_gpu(filter_col*filter_channel, weight);
  random_normal_gpu(filter_channel, bias);

  shared_size_ = out_w*out_h*filter_col*batch;
  input_size = batch*im_size;

#ifdef GEMMBITSERIAL
  ctx = allocGEMMContext(batch*out_h*out_w, filter_col, filter_channel, 1, 1, 1, 1);
#else
  bitset_outcol = new Bitset[batch*out_h*out_w];
  bitset_weight = new Bitset[filter_channel];

  for(int i = 0; i < batch*out_h*out_w; i++)
    bitset_outcol[i].Init(filter_col);

  for(int i = 0; i < filter_channel; i++)
    bitset_weight[i].Init(filter_col);
#endif

  shared_size_ = out_w*out_h*filter_col*batch;
  input_size = batch*im_size;

}

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

void BinaryConv::BinarizeInput() {

  int size = batch*height*width;
  input_mean_gpu_kernel<<<default_grid(size), BLOCK>>>(
      avg_filter,
      input,
      size,
      channel);
  check_error(cudaGetLastError());

  size = batch*out_h*out_w*filter_col;
  binarize_input_gpu_kernel<<<default_grid(size), BLOCK>>>(shared_, size);
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

void BinaryConv::BinarizeWeight() {

  weight_mean_gpu_kernel<<<default_grid(filter_channel), BLOCK>>>
    (mean, weight, filter_channel, filter_col);
  check_error(cudaGetLastError());

  int size = filter_col*filter_channel;
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

void BinaryConv::SwapWeight() {
    float *swap = weight;
    weight = binary_weight;
    binary_weight = swap;
}


void BinaryConv::Forward() {

  BinarizeWeight();
  SwapWeight();

  for(int i = 0; i < batch; i++) {
    im2col_gpu(
        width,
        height,
        channel,
        filter_width,
        filter_height,
        filter_channel,
        stride,
        pad,
        input + i*im_size,
        shared_ + i*col_size);
  }

  BinarizeInput();

  gemm_gpu(
      TRS_N,
      TRS_N,
      batch*out_h*out_w,
      filter_channel,
      filter_col,
      1, 
      shared_,
      weight, 
      output);


  // Do K = A (*) k
  int avg_filter_size = width*height;
  int avg_col_size = out_w*out_h*filter_width*filter_height;
  for(int i = 0; i < batch; i++) {
    im2col_gpu(
        width,
        height,
        1,
        filter_width,
        filter_height,
        1,
        stride,
        pad,
        avg_filter + i*avg_filter_size,
        avg_col + i*avg_col_size);
  }

  gemm_gpu(
    TRS_N,
    TRS_N,
    batch*out_h*out_w,
    1.0,
    filter_width*filter_height,
    1.0,
    avg_col,
    k_filter,
    k_output);

  int size = batch*out_w*out_h*filter_channel;
  multi_mean_gpu_kernel<<<default_grid(size), BLOCK>>>(output, mean, k_output, size, filter_channel);

  SwapWeight();
  BiasAdd();
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

void BinaryConv::BiasAdd() {

  size_t size = out_w*out_h*batch*filter_channel;
  bias_add_kernel<<<default_grid(size), BLOCK>>>(output, bias, out_w, out_h, filter_channel, size);
  check_error(cudaGetLastError());
}


__global__ void full_weight_mean_gpu_kernel(float *mean, float *weight, float *weight_binary, int n, int c) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i >= n) return;
  mean[i] = 0.0;
  for(int j = 0; j < c; j++)
    mean[i] += fabs(weight[i*c+j]);
  mean[i] /= (float)c;

  for(int j = 0; j < c; j++)
    weight_binary[i*c+j] = (weight[i*c+j] > 0 ) ? mean[i] : -mean[i];
}


void BinaryConv::Backward(float *delta) {

  for(int i = 0; i < batch; i++) {
    im2col_gpu(
        width,
        height,
        channel,
        filter_width,
        filter_height,
        filter_channel,
        stride,
        pad,
        input + i*im_size,
        shared_ + i*col_size);
  }

  gemm_gpu(
      TRS_T,
      TRS_N,
      filter_col,
      filter_channel,
      out_w*out_h*batch,
      1.0,
      shared_,
      delta,
      grad_weight);

  row_sum_gpu(batch*out_w*out_h, filter_channel, delta, grad_bias);

  full_weight_mean_gpu_kernel<<<default_grid(filter_channel), BLOCK>>>(
      mean,
      weight,
      binary_weight,
      filter_channel,
      filter_col);

  gemm_gpu(
      TRS_N,
      TRS_T,
      batch*out_w*out_h,
      filter_col,
      filter_channel,
      1.0,
      delta,
      binary_weight,
      shared_);

  for(int i = 0; i < batch; i++) {
    col2im_gpu(
        width,
        height,
        channel,
        filter_width,
        filter_height,
        filter_channel,
        stride,
        pad,
        delta_ + i*im_size,
        shared_  + i*col_size);
  }

}

void BinaryConv::Update(UpdateArgs update_args) {

  axpy_gpu(filter_col*filter_channel, update_args.decay, weight, grad_weight);

  if(update_args.adam) {
    adam_gpu(filter_col*filter_channel, weight, grad_weight, m_weight, v_weight, update_args);
    adam_gpu(filter_channel, bias, grad_bias, m_bias, v_bias, update_args);
  }
  else {
    momentum_gpu(filter_col*filter_channel, weight, grad_weight, v_weight, update_args);
    momentum_gpu(filter_channel, bias, grad_bias, v_bias, update_args);
  }
}
