#include "layers.h"

AvgPool::AvgPool(int _W, int _H, int _C,
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

AvgPool::~AvgPool() {

}

void AvgPool::init() {

#ifdef GPU
  col = malloc_gpu(out_w*out_h*out_channel);
  output = malloc_gpu(batch*out_w*out_h*FC);
  out_col = malloc_gpu(out_w*out_h*out_channel*batch);
  im = malloc_gpu(H*W*C);
  m_delta = malloc_gpu(batch*H*W*C);
  delta_col = malloc_gpu(batch*out_channel*out_w*out_h);
  indexes = malloc_gpu(batch*out_w*out_h*FC);
#else
  col = new float[out_w*out_h*out_channel];
  output = new float[batch*out_w*out_h*FC];
  out_col = new float[out_w*out_h*out_channel*batch];
  im = new float[H*W*C];
  m_delta = new float[batch*H*W*C];
  delta_col = new float[batch*out_channel*out_w*out_h];
  indexes = new float[batch*out_w*out_h*FC];
#endif

}


void AvgPool::forward() {

//ms_t s = getms();

#ifdef GPU
  forward_gpu();
#else
  int out_w = (W + 2*pad - FW)/stride + 1;
  int out_h = (H + 2*pad - FH)/stride + 1;
  int offset_w = 0, offset_h = 0;

  for(int b = 0; b < batch; b++) {
    for(int i = 0; i < out_h; i++) {
      for(int j = 0; j < out_w; j++) {

        for(int k = 0; k < FC; k++) {

	  float max = -1.0e+6;
	  int max_idx = -1;
          for(int n = 0; n < FH; n++) { 
            for(int m = 0; m < FW; m++) {
              int im_row = i*stride + n;
              int im_col = j*stride + m;
	      int idx = b*H*W*C + im_row*W*C + im_col*C + k;
              max_idx = (input[idx] > max) ? idx : max_idx;
              max = (input[idx] > max) ? input[idx] : max;
            }
          }

          int out_idx = b*out_h*out_w*FC + i*out_w*FC + j*FC + k;
          output[out_idx] = max;
          indexes[out_idx] = max_idx;
	}
      }
    }
  }

#endif
//cout << getms() - s << endl;
}

void AvgPool::backward(float *delta) {

#ifdef GPU
  backward_gpu(delta);
#else
  for(int i = 0; i < out_w*out_h*FC*batch; i++) {
    int j = indexes[i];
    m_delta[j] = delta[i];
  }
#endif

}

void AvgPool::update(update_args a) {
}

void AvgPool::save(fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "AvgPool,%d,%d,%d,%d,%d,%d,%d,%d",
    W, H, C, FW, FH, FC, stride, pad);
  //cout << buf << endl;
  file->write(buf, sizeof(buf));
}


AvgPool* AvgPool::load(char* buf) {

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

  AvgPool *pool = new AvgPool(para[0], para[1],
  para[2], para[3], para[4], para[5], para[6], para[7]);
  return pool;
}


