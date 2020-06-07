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

}


__global__ void bin_active_gpu_kernel(float *input, int n) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i >= n) return;
  input[i] = (input[i] > 0) ? 1.0 : -1.0;
}

void BinaryConv::BinActive() {

  size_t n = batch*out_w*out_h*filter_col;
  bin_active_gpu_kernel<<<default_grid(n), BLOCK>>>(shared_, n);
  check_error(cudaGetLastError());

}

__global__ void binarize_weight_gpu_kernel(float *weight, float *binary_weight,
		                           int n, int c) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i >= n) return;

  float mean = 0.0;

  for(int j = 0; j < c; j++)
    mean += fabs(weight[j*n+i]);
  mean /= (float)c;

  for(int j = 0; j < c; j++)
    binary_weight[j*n+i] = (weight[j*n+i] > 0 ) ? mean : -mean;

}


void BinaryConv::BinarizeWeight() {
  binarize_weight_gpu_kernel<<<default_grid(filter_channel), BLOCK>>>(
      weight,
      binary_weight,
      filter_channel,
      filter_col);

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

  BinActive();
  BinarizeWeight();

  gemm_gpu(
      TRS_N,
      TRS_N,
      batch*out_h*out_w,
      filter_channel,
      filter_col,
      1, 
      shared_,
      binary_weight, 
      output);

  bias_add_gpu(
      output,
      bias,
      batch*out_w*out_h*filter_channel,
      filter_channel);

}


__global__ void bin_active_backward_gpu_kernel(float *delta, float *input, int n) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;
  delta[i] = (fabs(input[i]) >= 1.0) ? 0 : delta[i];
}



void BinaryConv::BinActiveBackward() {

  size_t n = batch*width*height*channel;
  bin_active_backward_gpu_kernel<<<default_grid(n), BLOCK>>>(delta_, input, n);
  check_error(cudaGetLastError());

}


__global__ void update_grad_weight_gpu_kernel(float *grad_weight, float *weight,
    float *binary_weight, int n, int filter_col) {

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i >= n) return;

  float tmp;
  tmp = (fabs(weight[i]) >= 1.0) ? 0 : weight[i];
  grad_weight[i] = grad_weight[i]*(1.0/(float)(filter_col) + tmp*fabs(binary_weight[i]));
	  
}



void BinaryConv::UpdateGradientWeight() {

  update_grad_weight_gpu_kernel<<<default_grid(filter_channel*filter_col), BLOCK>>>(
      grad_weight,
      weight,
      binary_weight,
      filter_channel*filter_col,
      filter_col);
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

  BinActive();

  gemm_gpu(
      TRS_T,
      TRS_N,
      filter_col,
      filter_channel,
      batch*out_w*out_h,
      1.0,
      shared_,
      delta,
      grad_weight);

  UpdateGradientWeight();

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

  BinActiveBackward();

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

void BinaryConv::Save(std::fstream *file) {

  char buf[64] = {0};
  sprintf(buf, "BinaryConv,%d,%d,%d,%d,%d,%d,%d,%d",
    width, height, channel, filter_width, filter_height, filter_channel, stride, pad);
  file->write(buf, sizeof(buf));
 std::cout << buf<< std::endl;
  float *weight_tmp = new float[weight_size];
  gpu_pull_array(weight, weight_tmp, weight_size);
  file->write((char*)weight_tmp, weight_size*sizeof(float));
  delete []weight_tmp;

  float *bias_tmp = new float[bias_size];
  gpu_pull_array(bias, bias_tmp, bias_size);
  file->write((char*)bias_tmp, bias_size*sizeof(float));
  delete []bias_tmp;

}

void BinaryConv::LoadParams(std::fstream *file, int batch) {}
