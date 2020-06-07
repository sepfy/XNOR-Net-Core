#include "layer/activation.h"

const char *kActivationName[] = {
  "Relu",
  "Leaky",
  "Sigmoid"
};

void Activation::Print() {
  printf("%s\n", kActivationName[activation_type_]);
}

void Activation::Save(std::fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Activation,%d,%d", N, activation_type_);
  //cout << buf << endl;
  file->write(buf, sizeof(buf));
}

Activation* Activation::load(char *buf) {

  int para[2] = {0};
  char *token;
  int idx = 0;
  while (buf) {
    token = strtok(NULL, ",");
    para[idx] = atoi(token);
    idx++;
    if(idx > 1)
      break;
  }

  Activation *actv = new Activation(para[0], (ActivationType)para[1]);
  return actv;
}

#ifndef GPU
void Activation::Init() {

  output = new float[batch*N];
  if(train_flag_) {
    delta_ = new float[batch*N];
    cut = new float[batch*N];
    memset(cut, 0, sizeof(float)*batch*N);
  }
}

void Activation::Forward() {

  switch(activation_type_) {
    case RELU:
      relu_activate();
      break;
    case LEAKY:
      leaky_activate();
      break;
    case SIGMD:
      sigmoid_activate();
      break;
  }

}

void Activation::relu_activate() {

  for(int i = 0; i < batch; i++)
    for(int j = 0; j < N; j++)
      output[i*N+j] = (input[i*N+j] >= 0 ? input[i*N+j] : 0);
}

void Activation::leaky_activate() {

  for(int i = 0; i < batch; i++)
    for(int j = 0; j < N; j++)
      output[i*N+j] = (input[i*N+j] >= 0 ? input[i*N+j] : 0.1*input[i*N+j]);
}

void Activation::sigmoid_activate() {

  for(int i = 0; i < batch; i++)
    for(int j = 0; j < N; j++)
      output[i*N+j] = 1.0/(1.0 + exp(-1.0*(input[i*N+j])));
}

void Activation::Backward(float *delta) {

  switch(activation_type_) {
    case RELU:
      relu_backward(delta);
      break;
    case LEAKY:
      leaky_backward(delta);
      break;
    case SIGMD:
      sigmoid_backward(delta);
      break;
  }

}

void Activation::relu_backward(float *delta) {
  for(int i = 0; i < batch; i++)
    for(int j = 0; j < N; j++)
      delta_[i*N+j] = (cut[i*N+j] + delta[i*N+j])*(input[i*N+j] >= 0);
}

void Activation::leaky_backward(float *delta) {
  for(int i = 0; i < batch; i++)
    for(int j = 0; j < N; j++)
      delta_[i*N+j] = (cut[i*N+j] + delta[i*N+j])*(input[i*N+j] >= 0 ? 1.0 : 0.1);
}

void Activation::sigmoid_backward(float *delta) {
  for(int i = 0; i < batch; i++)
    for(int j = 0; j < N; j++)
      delta_[i*N+j] = (cut[i*N+j] + delta[i*N+j])*(1.0 - output[i*N+j])*output[i*N+j];
}

void Activation::LoadParams(std::fstream *file, int batch) {}

#endif
