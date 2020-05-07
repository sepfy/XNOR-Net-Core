#include "dropout.h"

Dropout::Dropout(int N, float ratio) {
  this->N = N;
  this->ratio = ratio;
}

Dropout::~Dropout() {

}


void Dropout::Init() {

#ifdef GPU
  output = malloc_gpu(batch*N);
  delta_ = malloc_gpu(batch*N);
  mask = malloc_gpu(batch*N);
  prob = malloc_gpu(batch*N);
#else	
  output = new float[batch*N];
  delta_ = new float[batch*N];
  mask = new float[batch*N];
#endif

}

void Dropout::Print() {
  printf("Dropout\n");
}

void Dropout::Forward() {

  if(train_flag_) {
    srand(time(NULL));
    for(int i = 0; i < batch; i++) 
      for(int j = 0; j < N; j++) {
        float prob = (float)rand()/(RAND_MAX + 1.0);
        mask[i*N+j] = (prob > ratio ? 1.0 : 0.0);
        output[i*N+j] = input[i*N+j]*mask[i*N+j];
      }
  }
  else {
    for(int i = 0; i < batch; i++) 
      for(int j = 0; j < N; j++) 
        output[i*N+j] = input[i*N+j];
  }
      
      //output[i*N+j] = (input[i*N+j] >= 0 ? input[i*N+j] : 0);
}

void Dropout::Backward(float *delta) {

  for(int i = 0; i < batch; i++) 
    for(int j = 0; j < N; j++)
      delta_[i*N+j] = delta[i*N+j]*mask[i*N+j];

 
      //delta_[i*N+j] = delta[i*N+j]*(input[i*N+j] >= 0);
}


void Dropout::Save(std::fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Dropout,%d", N);
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
