#include "dropout.h"

void Dropout::Print() {
  printf("Dropout\n");
}

#ifndef GPU

void Dropout::Init() {
  output = new float[batch*n_];
  delta_ = new float[batch*n_];
  mask_ = new float[batch*n_];
}

void Dropout::Forward() {

  if(train_flag_) {
    srand(time(NULL));
    for(int i = 0; i < batch; i++) 
      for(int j = 0; j < n_; j++) {
        float prob = (float)rand()/(RAND_MAX + 1.0);
        mask_[i*n_+j] = (prob > ratio_ ? 1.0 : 0.0);
        output[i*n_+j] = input[i*n_+j]*mask_[i*n_+j];
      }
  }
  else {
    for(int i = 0; i < batch; i++) 
      for(int j = 0; j < n_; j++) 
        output[i*n_+j] = input[i*n_+j];
  }

}

void Dropout::Backward(float *delta) {

  for(int i = 0; i < batch; i++) 
    for(int j = 0; j < n_; j++)
      delta_[i*n_+j] = delta[i*n_+j]*mask_[i*n_+j];
}

#endif

void Dropout::Save(std::fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Dropout,%d", n_);
  //cout << buf << endl;
  file->write(buf, sizeof(buf));
}

Dropout* Dropout::load(char *buf) {

  int para = 0;
  char *token;
  token = strtok(NULL, ",");
  para = atoi(token);
  Dropout *dropout = new Dropout(para, 0);
  return dropout;
}

void Dropout::LoadParams(std::fstream *file, int batch) {}
