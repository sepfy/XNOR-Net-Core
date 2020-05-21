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

__global__ void fast_binarize_gpu_kernel(float *x, float *x_binary,
		                         int n, int c, bool transpose) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i >= n) return;

  float mean = 0.0;

  if(transpose) {

    for(int j = 0; j < c; j++)
      mean += fabs(x[j*n+i]);
    mean /= (float)c;

    for(int j = 0; j < c; j++)
      x_binary[j*n+i] = (x[j*n+i] > 0 ) ? mean : -mean;
  }
  else {

    for(int j = 0; j < c; j++)
      mean += fabs(x[i*c+j]);
    mean /= (float)c;

    for(int j = 0; j < c; j++)
      x_binary[i*c+j] = (x[i*c+j] > 0 ) ? mean : -mean;

  }

}

void BinaryConv::BinarizeInput() {
  fast_binarize_gpu_kernel<<<default_grid(out_w*out_h), BLOCK>>>(
      shared_,
      shared_,
      out_w*out_h,
      filter_col,
      false);
  check_error(cudaGetLastError());
}


void BinaryConv::BinarizeWeight() {
  fast_binarize_gpu_kernel<<<default_grid(filter_channel), BLOCK>>>(
      weight,
      binary_weight,
      filter_channel,
      filter_col,
      true);
  check_error(cudaGetLastError());
}


void BinaryConv::Forward() {

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
  BinarizeWeight();

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

  BiasAdd();
}


__global__ void bias_add_kernel(float *output, float *bias, int n, int c) {

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;

    int k = i%c;

    output[i] += bias[k];

}

void BinaryConv::BiasAdd() {

  size_t n = batch*out_w*out_h*filter_channel;
  bias_add_kernel<<<default_grid(n), BLOCK>>>(output, bias, n, filter_channel);
  check_error(cudaGetLastError());
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

  BinarizeInput();

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
