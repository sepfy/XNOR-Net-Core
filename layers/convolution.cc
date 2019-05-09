#include "layers.h"

Convolution::Convolution(int _batch, int _W, int _H, int _C,
  int _FW, int _FH, int _FC, int _stride, int _pad, float* _input) {

  batch = _batch;
  W = _W;
  H = _H;
  C = _C;
  FW = _FW;
  FH = _FH;
  FC = _FC;
  stride = _stride;
  pad = _pad;
  out_w = (W + 2*pad - FW)/stride + 1;
  out_h = (H + 2*pad - FH)/stride + 1;
  out_channel = FW*FH*C;
  input = _input;
  init();

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
  // initialize weight with random number
  srand(time(NULL));
  for(int i = 0; i < out_channel; i++) {
    for(int j = 0; j < FC; j++) {
      weight[i*FC+j] = 0.1*((float) rand()/(RAND_MAX + 1.0) -0.5);
    }
  }

  for(int i = 0; i < out_w*out_h*FC; i++)
    bias[i] = 0.1*((float) rand()/(RAND_MAX + 1.0) -0.5);

}


void Convolution::forward() {
  int col_size = out_w*out_h*out_channel;
  int im_size = H*W*C;

  //  im = H*W*C
  // col = (out_h*out_w)*(out_channel)
  for(int i = 0; i < batch; i++) {
    memcpy(im, input + i*im_size, im_size*sizeof(float));
    im2col(W, H, C, FW, FH, FC, stride, pad, im, col);
    memcpy(out_col + i*col_size, col, col_size*sizeof(float));
  }

  // out_col = (batch*out_h*out_w)*out_channel
  // weight  = out_channel*FC
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


  int im_size = W*H*C;
  int col_size = out_w*out_h*out_channel;
 
  // weight = out_channel*FC
  // delta = batch*out_w*out_h*FC
  float *delta_col = new float[batch*out_channel*out_w*out_h];
  memset(delta_col, 0, batch*out_channel*out_w*out_h*sizeof(float));
  gemm_tb(batch*out_w*out_h, out_channel, FC, 1.0, delta, weight, delta_col);
  //delta_col = batch*out_w*out_h*out_channel 

  memset(m_delta, 0, batch*W*H*C*sizeof(float));

  for(int i = 0; i < batch; i++) {
    memcpy(col, delta_col + i*col_size, col_size*sizeof(float));
    col2im(W, H, C, FW, FH, FC, stride, pad, im, col);
    memcpy(m_delta + i*im_size, im, im_size*sizeof(float));
  }

  free(delta_col);



}

void Convolution::update() {

  //weight
  mat_scalar(out_channel, FC, grad_weight, 0.1, grad_weight);
  mat_minus(out_channel, FC, weight, grad_weight, weight);

  // bias
  mat_scalar(1, out_w*out_h*FC, grad_bias, 0.1, grad_bias);
  mat_minus(1, out_w*out_h*FC, bias, grad_bias, bias);

}

