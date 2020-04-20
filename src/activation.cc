#include "layers.h"


SoftmaxWithCrossEntropy::SoftmaxWithCrossEntropy(int n) {
  N = n;
}

void SoftmaxWithCrossEntropy::init() {

#ifdef GPU
  output = malloc_gpu(batch*N);
  m_delta = malloc_gpu(batch*N);
#else
  output = new float[batch*N];
  m_delta = new float[batch*N];
#endif

}

SoftmaxWithCrossEntropy::~SoftmaxWithCrossEntropy() {

#ifdef GPU
#else
  delete []output;
  delete []m_delta;
#endif

}    

void SoftmaxWithCrossEntropy::print() {

  float umem = (float)(2*batch*N)/(1024*1024);
  printf("Softmax\t %.2f\n", umem);
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

void SoftmaxWithCrossEntropy::update(update_args a) {
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



Activation::Activation(int N, ACT act) {
  this->N = N;
  activation = act;
}

Activation::~Activation() {

#ifdef GPU
#else
  delete []output;
  if(train_flag) {
    delete []m_delta;
    delete []m_delta;
  }
#endif


}

void Activation::init() {

#ifdef GPU
  output = malloc_gpu(batch*N);
  m_delta = malloc_gpu(batch*N);
  cut = malloc_gpu(batch*N);
#else
  output = new float[batch*N];
  if(train_flag) {
    m_delta = new float[batch*N];
    cut = new float[batch*N];
    memset(cut, 0, sizeof(float)*batch*N);
  }
#endif

}

string act_table[NUM_TYPE] = {"Relu", "Leaky", "SIGM"};

void Activation::print() {
  float umem = (float)(3*batch*N)/(1024*1024);
  printf("%s \t %.2f\n", act_table[activation].c_str(), umem);
}


void Activation::forward() {

  switch(activation) {
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


void Activation::backward(float *delta) {

  switch(activation) {
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
      m_delta[i*N+j] = (cut[i*N+j] + delta[i*N+j])*(input[i*N+j] >= 0);
}

void Activation::leaky_backward(float *delta) {
  for(int i = 0; i < batch; i++)
    for(int j = 0; j < N; j++)
      m_delta[i*N+j] = (cut[i*N+j] + delta[i*N+j])*(input[i*N+j] >= 0 ? 1.0 : 0.1);
}

void Activation::sigmoid_backward(float *delta) {
  for(int i = 0; i < batch; i++)
    for(int j = 0; j < N; j++)
      m_delta[i*N+j] = (cut[i*N+j] + delta[i*N+j])*(1.0 - output[i*N+j])*output[i*N+j];
}

void Activation::update(update_args a) {
}

void Activation::save(fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Activation,%d,%d", N, activation);
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

  Activation *actv = new Activation(para[0], (ACT)para[1]);
  return actv;
}
