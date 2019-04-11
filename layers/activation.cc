#include "layers.h"

Sigmoid::Sigmoid(int _batch, int _N, float *_input) {
  batch = _batch;
  N = _N;
  input = _input;
  init();  
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

SoftmaxWithCrossEntropy::SoftmaxWithCrossEntropy(int _batch,
    int _n, float *_target, float *_input) {
  batch = _batch;
  N = _n;
  target = _target;
  input = _input;
  init();
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
  mat_minus(batch, N, output, target, m_delta);  
  mat_scalar(batch, N, m_delta, 1.0/(float)batch, m_delta);
}


Relu::Relu(int _batch, int _N, float *_input) {
  batch = _batch;
  N = _N;
  input = _input;
}

Relu::~Relu() {

}

void Relu::init() {
  output = new float[batch*N];
}

void Relu::forward() {

  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      output[i*N+j] = (input[i*N+j] >= 0 ? input[i*N+j] : 0);
    }
  }
}

void Relu::backward(float *delta) {

}

