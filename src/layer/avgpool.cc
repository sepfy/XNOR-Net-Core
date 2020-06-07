#include "layer/avgpool.h"

Avgpool::Avgpool(int W, int H, int C,
  int FW, int FH, int FC, int stride, bool pad) {

  this->W = W;
  this->H = H;
  this->C = C;
  this->FW = FW;
  this->FH = FH;
  this->FC = FC;
  this->stride = stride;

}


void Avgpool::Print() {
  printf("Avgpool \t %d x %d x %d \t\t %d x %d x %d \n", H, W, C, 1, 1, FC);
}


void Avgpool::LoadParams(std::fstream *rfile, int batch) {}

#ifndef GPU
void Avgpool::Init() {

  output = new float[batch*C];
  if(train_flag_)
    delta_ = new float[batch*H*W*C];

}


void Avgpool::Forward() {

  memset(output, 0, sizeof(float)*batch*C);
  for(int b = 0; b < batch; b++)
    for(int n = 0; n < H; n++)
      for(int m = 0; m < W; m++)
        for(int k = 0; k < C; k++) {
          int out_idx = b*C + k;
          int idx = b*H*W*C + n*W*C + m*C + k;
          output[out_idx] += input[idx];
        }
  
  for(int b = 0; b < batch; b++)
    for(int k = 0; k < C; k++)
      output[b*C+k] /= (float)(W*H);
   
    

}

void Avgpool::Backward(float *delta) {

  for(int b = 0; b < batch; b++)
    for(int n = 0; n < H; n++)
      for(int m = 0; m < W; m++)
        for(int k = 0; k < C; k++) {
          int out_idx = b*C + k;
          int idx = b*H*W*C + n*W*C + m*C + k;
	  delta_[idx] = delta[out_idx]/(float)(H*W);
        }

}
#endif

void Avgpool::Save(std::fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Avgpool,%d,%d,%d,%d,%d,%d,%d,%d",
    W, H, C, FW, FH, FC, stride, pad);
  //cout << buf << endl;
  file->write(buf, sizeof(buf));
}


Avgpool* Avgpool::load(char* buf) {

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

  Avgpool *pool = new Avgpool(para[0], para[1],
  para[2], para[3], para[4], para[5], para[6], para[7]);
  return pool;
}


