#include "layers.h"

AvgPool::AvgPool(int W, int H, int C,
  int FW, int FH, int FC, int stride, bool pad) {

  this->W = W;
  this->H = H;
  this->C = C;
  this->FW = FW;
  this->FH = FH;
  this->FC = FC;
  this->stride = stride;

}

AvgPool::~AvgPool() {

}

void AvgPool::init() {

#ifdef GPU
  output = malloc_gpu(batch*C);
  m_delta = malloc_gpu(batch*H*W*C);
#else
  output = new float[batch*C];
  m_delta = new float[batch*H*W*C];
#endif

}

void AvgPool::print() {

  float umem = (float)(batch*C + batch*H*W*C)/(1024*1024);
  printf("Avg \t %.2f \t %d x %d x %d \t\t %d x %d x %d \n",
                  umem, H, W, C, 1, 1, FC);

	  
}


void AvgPool::forward() {

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

void AvgPool::backward(float *delta) {

  for(int b = 0; b < batch; b++)
    for(int n = 0; n < H; n++)
      for(int m = 0; m < W; m++)
        for(int k = 0; k < C; k++) {
          int out_idx = b*C + k;
          int idx = b*H*W*C + n*W*C + m*C + k;
	  m_delta[idx] = delta[out_idx]/(float)(H*W);
        }

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


