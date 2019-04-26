#include "layers.h"

Pooling::Pooling(int _batch, int _W, int _H, int _C,
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

Pooling::~Pooling() {

}

void Pooling::init() {
  col = new float[out_w*out_h*out_channel];
  output = new float[batch*out_w*out_h*FC];
  out_col = new float[out_w*out_h*out_channel*batch];
  im = new float[H*W*C];
  m_delta = new float[batch*out_channel*out_w*out_h];
}


void Pooling::forward() {
  int col_size = out_w*out_h*out_channel;
  int im_size = H*W*C;

  //  im = H*W*C
  // col = (out_h*out_w)*(out_channel)
  for(int i = 0; i < batch; i++) {
    memcpy(im, input + i*im_size, im_size*sizeof(float));
    im2col(W, H, C, FW, FH, C, stride, pad, im, col);
    //cout << endl;
    memcpy(out_col + i*col_size, col, col_size*sizeof(float));
  }
  int out_size = out_h*out_w*C;
  int filter_size = FH*FW;
  int idx;
  float max;
  float max_idx = 0;
  max_seq.clear();
  //out_channel = FW*FH*C;
  //col_size = (out_h*out_w)*(out_channel)
  for(int i = 0; i < batch; i++) {
    for(int p = 0; p < out_h; p++)
      for(int q = 0; q < out_w; q++)
        for(int o = 0; o < C; o++) {
          max = -1.0e+6;
          for(int j = 0; j < filter_size; j++) {
            idx = i*col_size + p*(out_w*out_channel) + q*out_channel + o*filter_size + j;
            //cout << out_col[idx] << " ";
            if(out_col[idx] > max) {
              //out_w*out_h*out_channel*batch
              max = out_col[idx];
              max_idx = idx;
            }
          }
          //cout << endl;
          output[i*out_size+p*(out_w*C)+q*C+o] = max;
          max_seq.push_back(max_idx);
        }
  }  
}

void Pooling::backward(float *delta) {
  //m_delta = delta;


  memset(m_delta, 0, batch*out_channel*out_w*out_h); 
  for(int i = 0; i < max_seq.size(); i++)
    m_delta[max_seq[i]] = delta[max_seq[i]];
/*
  int filter_size = FH*FW;
  for(int i = 0; i < batch; i++) 
    for(int p = 0; p < out_h; p++)
      for(int q = 0; q < out_w; q++)
        for(int o = 0; o < C; o++) 
          for(int j = 0; j < filter_size; j++) {
            int idx = i*out_h*out_w*C + p*out_w*C + q*C + o;
cout << max_seq[idx] << endl;
            if(j == max_seq[idx])
              m_delta[idx] = delta[idx];
            else
              m_delta[idx] = 0.0;
          }
     */ 
    

  //m_delta = delta;
  //memset(grad_weight, 0, out_channel*FC*sizeof(float));
  //memset(m_delta, 0, batch*N*sizeof(float));
  //grad_weight = out_channel*FC
  // out_col = (batch*out_h*out_w)*out_channel
  // delta      = (batch*out_h*out_w)*FC
  //gemm_ta(out_channel, FC, out_h*out_w*batch, 1.0, out_col, delta, grad_weight);
}

void Pooling::update() {

}

