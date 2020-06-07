#include "layer/convolution.h"
#include "utils.h"
#include "blas.h"
#include "gemm.h"

void Convolution::Print() {
  printf("Convolution \t %d x %d x %d \t\t %d x %d x %d \n",
      H, W, C, out_h, out_w, FC);

}

Convolution* Convolution::load(char *buf) {

  int para[9] = {0};
  int idx = 0;

  char *token;
  while (buf) {
    token = strtok(NULL, ",");
    para[idx] = atoi(token);
    idx++;
    if(idx > 8)
      break;
  }

  Convolution *conv = new Convolution(para[0], para[1], 
  para[2], para[3], para[4], para[5], para[6], para[7]);
  return conv;
}

#ifndef GPU

void Convolution::Init() {

  output = new float[batch*out_w*out_h*FC];
  weight = new float[out_channel*FC];
  bias = new float[FC];

  if(train_flag_) {
    grad_weight = new float[out_channel*FC];
    grad_bias = new float[FC];
    delta_ = new float[batch*W*H*C]; 

    /* Adam optimizer */
    m_weight = new float[out_channel*FC];
    v_weight = new float[out_channel*FC];
    m_bias = new float[FC];
    v_bias = new float[FC];

    random_normal(out_channel*FC, weight);
    random_normal(FC, bias);
    memset(m_weight, 0, out_channel*FC*sizeof(float));
    memset(v_weight, 0, out_channel*FC*sizeof(float));
    memset(m_bias, 0 , FC*sizeof(float));  
    memset(v_bias, 0 , FC*sizeof(float));  
  }


  shared_size_ = out_w*out_h*out_channel*batch;
  input_size = batch*im_size;
}



void Convolution::Forward() {

  for(int i = 0; i < batch; i++)
    im2col(W, H, C, FW, FH, FC, stride, pad, 
      input + i*im_size, shared_+i*col_size);

  gemm_cpu(TRS_N, TRS_N, batch*out_h*out_w, FC, out_channel, 1, shared_, weight, output);

  bias_add(output, bias, batch*out_h*out_w, FC);
}



void Convolution::Backward(float *delta) {

  for(int i = 0; i < batch; i++)
    im2col(W, H, C, FW, FH, FC, stride, pad,
      input + i*im_size, shared_+i*col_size);

  gemm_cpu(TRS_T, TRS_N, 
           out_channel, FC, out_h*out_w*batch, 1.0,
           shared_, delta, grad_weight);

  gemm_cpu(TRS_N, TRS_T,
         batch*out_w*out_h, out_channel, FC, 1.0,
         delta, weight, shared_);

  row_sum(batch*out_w*out_h, FC, delta, grad_bias);

  for(int i = 0; i < batch; i++)
    col2im(W,H, C, FW, FH, FC, stride, pad, 
      delta_ + i*im_size, shared_ + i*col_size);
}

void Convolution::Update(UpdateArgs update_args) {

  adam_cpu(out_channel*FC, weight, grad_weight, m_weight, v_weight, update_args);
  adam_cpu(FC, bias, grad_bias, m_bias, v_bias, update_args);

}

void Convolution::Save(std::fstream *file) {

  char buf[64] = {0};
  sprintf(buf, "Convolution,%d,%d,%d,%d,%d,%d,%d,%d", 
    W, H, C, FW, FH, FC, stride, pad);
  file->write(buf, sizeof(buf));
  file->write((char*)weight, weight_size*sizeof(float));
  file->write((char*)bias, bias_size*sizeof(float));
}

#endif


