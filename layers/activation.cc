#include "layers.h"

Sigmoid::Sigmoid(int _N) {
  N = _N;
}

Sigmoid::~Sigmoid() {

}

void Sigmoid::init() {
  output = new float[batch*N];
  m_delta = new float[batch*N];
}

void Sigmoid::forward() {
  for(int i = 0; i < batch; i++) 
    for(int j = 0; j < N; j++) 
      output[i*N+j] = 1.0/(1.0 + exp(-1.0*(input[i*N+j])));
}

void Sigmoid::backward(float *delta) {
  for(int i = 0; i < batch; i++) 
    for(int j = 0; j < N; j++) 
      m_delta[i*N+j] = delta[i*N+j]*(1.0 - output[i*N+j])*output[i*N+j];
}

void Sigmoid::update(float lr) {

}

void Sigmoid::save(fstream *file) {

}


SoftmaxWithCrossEntropy::SoftmaxWithCrossEntropy(int _n) {
  N = _n;
}

void SoftmaxWithCrossEntropy::init() {
  output = new float[batch*N];
  m_delta = new float[batch*N];
}

SoftmaxWithCrossEntropy::~SoftmaxWithCrossEntropy() {

}    

void SoftmaxWithCrossEntropy::forward() {

  for(int i = 0; i < batch; i++) {
    float tmp = 0;
    float max = 0;
    for(int j = 0; j < N; j++) 
      if(input[i*N+j] > max)
	max = input[i*N+j];

    for(int j = 0; j < N; j++) {
      output[i*N+j] = exp(input[i*N+j] - max);
      tmp += output[i*N+j];
    }
    for(int j = 0; j < N; j++) 
      output[i*N+j] /= tmp;
  }
}

void SoftmaxWithCrossEntropy::backward(float *delta) {
  mat_minus(batch, N, output, delta, m_delta);  
  mat_scalar(batch, N, m_delta, 1.0/(float)batch, m_delta);
}

void SoftmaxWithCrossEntropy::update(float lr) {
}

void SoftmaxWithCrossEntropy::save(fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Softmax,%d", N);
  //cout << buf << endl;
  file->write(buf, sizeof(buf));
}

SoftmaxWithCrossEntropy* SoftmaxWithCrossEntropy::load(char *buf) {
  int para = 0;
  char *token;
  token = strtok(NULL, ",");
  para = atoi(token);
  SoftmaxWithCrossEntropy *softmax= new SoftmaxWithCrossEntropy(para);
  return softmax;

}



Relu::Relu(int _N) {
  N = _N;
}

Relu::~Relu() {

}

void Relu::init() {
  output = new float[batch*N];
  m_delta = new float[batch*N];
}

void Relu::forward() {

  for(int i = 0; i < batch; i++) 
    for(int j = 0; j < N; j++) 
      output[i*N+j] = (input[i*N+j] >= 0 ? input[i*N+j] : 0);
}

void Relu::backward(float *delta) {

  for(int i = 0; i < batch; i++) 
    for(int j = 0; j < N; j++) 
      m_delta[i*N+j] = delta[i*N+j]*(input[i*N+j] >= 0);
}

void Relu::update(float lr) {
}

void Relu::save(fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Relu,%d", N);
  //cout << buf << endl;
  file->write(buf, sizeof(buf));
}

Relu* Relu::load(char *buf) {

  int para = 0;
  char *token;
  token = strtok(NULL, ",");
  para = atoi(token);
  Relu *relu = new Relu(para);
  return relu;
}
