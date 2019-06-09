#include "layers.h"
#include <bitset>

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

  random_normal(out_channel*FC, weight);
  random_normal(out_w*out_h*FC, bias);

}

#if XNOR_NET
void Convolution::binarize(float *input, int N) {

}
#endif


void Convolution::forward() {

  //binarize(input, im_size);
  for(int i = 0; i < batch; i++) {
    memcpy(im, input + i*im_size, im_size*sizeof(float));
    im2col(W, H, C, FW, FH, FC, stride, pad, im, col);
    memcpy(out_col + i*col_size, col, col_size*sizeof(float));
  }

  gemm(batch*out_h*out_w, FC, out_channel, 1, out_col, weight, output);
  bias_add(batch, out_h*out_w*FC, output, bias);
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
    memcpy(col, delta_col + i*col_size, col_size*sizeof(float));
    col2im(W, H, C, FW, FH, FC, stride, pad, im, col);
    memcpy(m_delta + i*im_size, im, im_size*sizeof(float));
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

