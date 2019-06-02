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

void Sigmoid::update() {

}

SoftmaxWithCrossEntropy::SoftmaxWithCrossEntropy(
    int _n, float *_target) {
  N = _n;
  target = _target;
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

void SoftmaxWithCrossEntropy::update() {
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

  int idx;
  mask.clear();
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      //output[i*N+j] = (input[i*N+j] >= 0 ? input[i*N+j] : 0);
      idx = i*N + j;
      if(input[idx] > 0) 
        output[idx] = input[idx];
      else {
        output[idx] = 0.0;
        mask.push_back(idx);
      }
    }
  }
}

void Relu::backward(float *delta) {


  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      m_delta[i*N+j] = delta[i*N+j];
    }
  }

  //copy(m_delta, m_delta + N*batch, delta);
  for(int i = 0; i < mask.size(); i++) {
    //delta[mask[i]] = 0.0;
    m_delta[mask[i]] = 0.0;
  }

}

void Relu::update() {
}
