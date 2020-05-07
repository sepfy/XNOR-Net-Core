#include "layer/maxpool.h"

void Maxpool::Print() {

  float umem = (float)(batch*out_w*out_h*FC*3)/(1024*1024);
  printf("Max \t %.2f \t %d x %d x %d \t %d x %d x %d \n",
                  umem, H, W, C, out_h, out_w, FC);
}

#ifndef GPU
void Maxpool::Init() {

  output = new float[batch*out_w*out_h*FC];
  if(train_flag_) {
    indexes = new float[batch*out_w*out_h*FC];
    delta_ = new float[batch*H*W*C];
  }

}



void Maxpool::Forward() {

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
	  if(train_flag_)
            indexes[out_idx] = max_idx;
	}
      }
    }
  }

}

void Maxpool::Backward(float *delta) {

  for(int i = 0; i < out_w*out_h*FC*batch; i++) {
    int j = indexes[i];
    delta_[j] = delta[i];
  }

}

#endif

void Maxpool::Save(std::fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Pooling,%d,%d,%d,%d,%d,%d,%d,%d",
    W, H, C, FW, FH, FC, stride, pad);
  //cout << buf << endl;
  file->write(buf, sizeof(buf));
}


Maxpool* Maxpool::load(char* buf) {

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

  Maxpool *pool = new Maxpool(para[0], para[1],
  para[2], para[3], para[4], para[5], para[6], para[7]);
  return pool;
}


