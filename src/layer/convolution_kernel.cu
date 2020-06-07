#include "layer/convolution.h"
#include "gemm.h"
#include "utils.h"
#include "blas.h"

void Convolution::Init() {

  output = malloc_gpu(batch*out_w*out_h*FC);
  weight = malloc_gpu(out_channel*FC);
  grad_weight = malloc_gpu(out_channel*FC);
  bias = malloc_gpu(FC);
  grad_bias = malloc_gpu(FC);
  delta_ = malloc_gpu(batch*W*H*C);

  // Adam optimizer
  m_weight = malloc_gpu(out_channel*FC);
  v_weight = malloc_gpu(out_channel*FC);
  m_bias = malloc_gpu(FC);
  v_bias = malloc_gpu(FC);

  random_normal_gpu(out_channel*FC, weight);
  random_normal_gpu(FC, bias);

  shared_size_ = out_w*out_h*out_channel*batch;
  input_size = batch*im_size;

  shared_size_ = out_w*out_h*out_channel*batch;
  input_size = batch*im_size;

}

void Convolution::Forward() {
 
  for(int i = 0; i < batch; i++)
    im2col_gpu(W, H, C, FW, FH, FC, stride, pad,
      input + i*im_size, shared_+i*col_size);

  gemm_gpu(TRS_N, TRS_N, batch*out_h*out_w, FC, out_channel, 1, shared_, weight, output);

  bias_add_gpu(
      output,
      bias,
      batch*out_w*out_h*FC,
      FC);
}


void Convolution::Backward(float *delta) {

  for(int i = 0; i < batch; i++)
    im2col_gpu(W, H, C, FW, FH, FC, stride, pad,
      input + i*im_size, shared_+i*col_size);

  gemm_gpu(TRS_T, TRS_N,
           out_channel, FC, out_h*out_w*batch, 1.0,
           shared_, delta, grad_weight);

  row_sum_gpu(batch*out_w*out_h, FC, delta, grad_bias);

  gemm_gpu(TRS_N, TRS_T,
       batch*out_w*out_h, out_channel, FC, 1.0,
       delta, weight, shared_);

  for(int i = 0; i < batch; i++) {
    col2im_gpu(W, H, C, FW, FH, FC, stride, pad,
      delta_ + i*im_size, shared_  + i*col_size);
  }
}

void Convolution::Update(UpdateArgs update_args) {

  axpy_gpu(out_channel*FC, update_args.decay, weight, grad_weight);

  if(update_args.adam) {
    adam_gpu(out_channel*FC, weight, grad_weight, m_weight, v_weight, update_args);
    adam_gpu(FC, bias, grad_bias, m_bias, v_bias, update_args);
  }
  else {
    momentum_gpu(out_channel*FC, weight, grad_weight, v_weight, update_args);
    momentum_gpu(FC, bias, grad_bias, v_bias, update_args);
  }
}


void Convolution::Save(std::fstream *file) {

  char buf[64] = {0};
  sprintf(buf, "Convolution,%d,%d,%d,%d,%d,%d,%d,%d",
    W, H, C, FW, FH, FC, stride, pad);
  file->write(buf, sizeof(buf));

  float *weight_tmp = new float[weight_size];
  gpu_pull_array(weight, weight_tmp, weight_size);
  file->write((char*)weight_tmp, weight_size*sizeof(float));
  delete []weight_tmp;

  float *bias_tmp = new float[bias_size];
  gpu_pull_array(bias, bias_tmp, bias_size);
  file->write((char*)bias_tmp, bias_size*sizeof(float));
  delete []bias_tmp;

}

void Convolution::LoadParams(std::fstream *rfile, int batch) {

  batch = batch;
  train_flag_ = false;
  runtime = true;
  Init();

  float *weight_tmp = new float[weight_size];
  rfile->read((char*)weight_tmp, weight_size*sizeof(float));
  gpu_push_array(weight, weight_tmp, weight_size);
  delete []weight_tmp;
  float *bias_tmp = new float[bias_size];
  rfile->read((char*)bias_tmp, bias_size*sizeof(float));
  gpu_push_array(bias, bias_tmp, bias_size);
  delete []bias_tmp;

}
