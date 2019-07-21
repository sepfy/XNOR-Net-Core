#include "layers.h"

Pooling::Pooling(int _W, int _H, int _C,
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
}

Pooling::~Pooling() {

}

void Pooling::init() {

  col = new float[out_w*out_h*out_channel];
  output = new float[batch*out_w*out_h*FC];
  out_col = new float[out_w*out_h*out_channel*batch];
  im = new float[H*W*C];
  m_delta = new float[batch*H*W*C];
  delta_col = new float[batch*out_channel*out_w*out_h];
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
            //idx = i*col_size + p*(out_w*out_channel) + q*out_channel + o*filter_size + j;
            idx = i*col_size + p*(out_w*out_channel) + q*out_channel + j*C + o;
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


  //im2col is batch*(H*W*C) => batch*(out_channel*out_w*out_h*FC)
  //col2im is batch*(out_channel*out_w*out_h*FC) => batch*(H*W*C)
  // (batch*out_w*out_h)*out_channel = FW*FH*FC 
  //m_delta should be (batch*out_h*out_w)*FC

  int im_size = W*H*C;
  int col_size = out_w*out_h*out_channel;
  float *im = new float[W*H*C];
  float *col = new float[out_channel*out_w*out_h];
  memset(delta_col, 0.0, batch*out_channel*out_w*out_h); 
  for(int i = 0; i < out_w*out_h*C*batch; i++)
    delta_col[max_seq[i]] = delta[i];
    //m_delta[max_seq[i]] = delta[i];
  //float
  for(int i = 0; i < batch; i++) {
    memcpy(col, delta_col + i*col_size, col_size*sizeof(float)); 
    col2im(W, H, C, FW, FH, C, stride, pad, im, col);
    memcpy(m_delta + i*im_size, im, im_size*sizeof(float));
  }
  free(im);
  
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

void Pooling::update(float lr) {
}

void Pooling::save(fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Pooling,%d,%d,%d,%d,%d,%d,%d,%d",
    W, H, C, FW, FH, FC, stride, pad);
  //cout << buf << endl;
  file->write(buf, sizeof(buf));
}


Pooling* Pooling::load(char* buf) {

  int para[8] = {0};
  int idx = 0;

  char *token;
  while (buf) {
    token = strtok(NULL, ",");
    para[idx] = atoi(token);
    idx++;
    if(idx > 7)
      break;
  }

  Pooling *pool = new Pooling(para[0], para[1],
  para[2], para[3], para[4], para[5], para[6], para[7]);
  return pool;
}


