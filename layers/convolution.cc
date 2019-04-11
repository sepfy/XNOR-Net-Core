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

  im = new float[H*W*C];

  // initialize weight with random number
  srand(time(NULL));
  for(int i = 0; i < out_channel; i++) {
    for(int j = 0; j < FC; j++) {
      weight[i*FC+j] = 0.1*((float) rand()/(RAND_MAX + 1.0) -0.5);
    }
  }
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
}

void Convolution::backward(float *delta) {

  memset(grad_weight, 0, out_channel*FC*sizeof(float));
  //memset(m_delta, 0, batch*N*sizeof(float));
  //grad_weight = out_channel*FC
  // out_col = (batch*out_h*out_w)*out_channel
  // delta      = (batch*out_h*out_w)*FC
  gemm_ta(out_channel, FC, out_h*out_w*batch, 1.0, out_col, delta, grad_weight);
}

void Convolution::update() {
  mat_scalar(out_channel, FC, grad_weight, 0.1, grad_weight);
  mat_minus(out_channel, FC, weight, grad_weight, weight);

}

