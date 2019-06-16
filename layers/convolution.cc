#include "layers.h"

using namespace std;

Convolution::Convolution(int _W, int _H, int _C,
  int _FW, int _FH, int _FC, int _stride, bool _pad) {

  W = _W;
  H = _H;
  C = _C;
  FW = _FW;
  FH = _FH;
  FC = _FC;
  stride = _stride;


  if(_pad == true) {
    pad = 0.5*((stride - 1)*W - stride + FW);
    out_w = W;
    out_h = H;
  }
  else {
    pad = 0;
    out_w = (W - FW)/stride + 1;
    out_h = (H - FH)/stride + 1;
  }

  out_channel = FW*FH*C;
  col_size = out_w*out_h*out_channel;
  im_size = H*W*C;
  weight_size = out_channel*FC;
  input_size = batch*im_size;
}

Convolution::~Convolution() {

}

void Convolution::init() {

  col = new float[out_w*out_h*out_channel];
  output = new float[batch*out_w*out_h*FC];
  out_col = new float[out_w*out_h*out_channel*batch];

  weight = new float[out_channel*FC];
  grad_weight = new float[out_channel*FC];
  bias = new float[out_w*out_h*FC];
  grad_bias = new float[out_w*out_h*FC];
  im = new float[H*W*C];
  m_delta = new float[batch*W*H*C]; 

#ifdef XNOR_NET
  binary_weight = new float[out_channel*FC];
#endif

  random_normal(out_channel*FC, weight);
  random_normal(out_w*out_h*FC, bias);

}

#ifdef XNOR_NET
void Convolution::swap_weight()
{
    float *swap = weight;
    weight = binary_weight;
    binary_weight = swap;
}

float Convolution::binarize_weight() {

  float l1 = 0.0;
  for(int i = 0; i < weight_size; i++) {
    l1 += fabs(weight[i]);
    weight[i] >= 0 ? binary_weight[i] = 1 : binary_weight[i] = -1;
  }
  return l1/(float)weight_size;
}

float Convolution::binarize_input() {
  float l1 = 0.0;
  for(int i = 0; i < input_size; i++) {
    l1 += fabs(input[i]);
    input[i] >= 0 ? input[i] = 1 : input[i] = -1;
  }
  return l1/(float)im_size;
}

#endif

void Convolution::forward() {

// I am not sure why this code is not working. It should be equivalent below!
#if 0
 for(int i = 0; i < batch; i++) {
    im2col(W, H, C, FW, FH, FC, stride, pad, input + i*im_size, col);
    gemm(out_h*out_w, FC, out_channel, 1, col, weight, output+i*out_h*out_w*FC);
  }
#endif

#ifdef XNOR_NET
//  binarize_input();
  for(int i = 0; i < batch; i++) {
    //memcpy(im, input + i*im_size, im_size*sizeof(float));
    im2col(W, H, C, FW, FH, FC, stride, pad, input + i*im_size, col);
    memcpy(out_col + i*col_size, col, col_size*sizeof(float));
  }
  float alpha = binarize_weight();
  swap_weight();
  gemm(batch*out_h*out_w, FC, out_channel, alpha, out_col, weight, output);

#else
  for(int i = 0; i < batch; i++) {
    //memcpy(im, input + i*im_size, im_size*sizeof(float));
    im2col(W, H, C, FW, FH, FC, stride, pad, input + i*im_size, col);
    memcpy(out_col + i*col_size, col, col_size*sizeof(float));
  }
  gemm(batch*out_h*out_w, FC, out_channel, 1, out_col, weight, output);
#endif
  bias_add(batch, out_h*out_w*FC, output, bias);

#ifdef XNOR_NET
  swap_weight();
#endif

}

void Convolution::backward(float *delta) {

  //weight
  memset(grad_weight, 0, out_channel*FC*sizeof(float));
  gemm_ta(out_channel, FC, out_h*out_w*batch, 1.0, out_col, delta, grad_weight);

  //bias
  memset(grad_bias, 0, out_w*out_h*FC*sizeof(float));
  row_sum(batch, out_w*out_h*FC, delta, grad_bias);

  float *delta_col = new float[batch*out_channel*out_w*out_h];
  gemm_tb(batch*out_w*out_h, out_channel, FC, 1.0, delta, weight, delta_col);

  for(int i = 0; i < batch; i++) {
    col2im(W,H, C, FW, FH, FC, stride, pad, m_delta + i*im_size, delta_col + i*col_size);
  }

  free(delta_col);

}

void Convolution::update(float lr) {

  //weight
  mat_scalar(out_channel, FC, grad_weight, lr, grad_weight);
  mat_minus(out_channel, FC, weight, grad_weight, weight);

  // bias
  mat_scalar(1, out_w*out_h*FC, grad_bias, lr, grad_bias);
  mat_minus(1, out_w*out_h*FC, bias, grad_bias, bias);

}

