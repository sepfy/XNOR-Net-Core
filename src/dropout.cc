#include "layers.h"

Dropout::Dropout(int _N, float _ratio) {
  N = _N;
  ratio = _ratio;
}

Dropout::~Dropout() {

}


void Dropout::init() {
  output = new float[batch*N];
  m_delta = new float[batch*N];
  mask = new float[batch*N];
}

void Dropout::forward() {

  if(train_flag) {
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

void Dropout::backward(float *delta) {

  for(int i = 0; i < batch; i++) 
    for(int j = 0; j < N; j++)
      m_delta[i*N+j] = delta[i*N+j]*mask[i*N+j];

 
      //m_delta[i*N+j] = delta[i*N+j]*(input[i*N+j] >= 0);
}

void Dropout::update(float lr) {
}

void Dropout::save(fstream *file) {
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
