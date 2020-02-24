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

void Sigmoid::update(update_args a) {

}

void Sigmoid::save(fstream *file) {

}


SoftmaxWithCrossEntropy::SoftmaxWithCrossEntropy(int _n) {
  N = _n;
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

}    

void SoftmaxWithCrossEntropy::forward() {

#ifdef GPU
  forward_gpu();
#else

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
#endif

}

void SoftmaxWithCrossEntropy::backward(float *delta) {

#ifdef GPU
  float alpha = 1.0/(float)batch;
  size_t size = sizeof(float)*batch*N;
  cudaError_t status = cudaMemset(m_delta, 0, size);
  check_error(status);

  cublasSaxpy(gpu_handle(), batch*N, &alpha, output, 1, m_delta, 1);    
  check_error(cudaGetLastError());

  alpha = -1.0/(float)batch;
  cublasSaxpy(gpu_handle(), batch*N, &alpha, delta, 1, m_delta, 1);    
  check_error(cudaGetLastError());
#else
  mat_minus(batch, N, output, delta, m_delta);  
  mat_scalar(batch, N, m_delta, 1.0/(float)batch, m_delta);
#endif
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



Relu::Relu(int _N, ACT act) {
  N = _N;
  activation = act;
}

Relu::~Relu() {

}

void Relu::init() {

#ifdef GPU
	
  output = malloc_gpu(batch*N);
  m_delta = malloc_gpu(batch*N);
  cut = malloc_gpu(batch*N);
   
#else
  output = new float[batch*N];
  m_delta = new float[batch*N];
  cut = new float[batch*N];
  memset(cut, 0, sizeof(float)*batch*N);
#endif

  cout << "Activation layers: memory = " << 3*batch*N/(1024*1024) << endl;
}

void Relu::forward() {


#ifdef GPU
   switch(activation) {
    case RELU:
      relu_activate_gpu();
    case LEAKY:
      leaky_activate_gpu();
  }
#else

  switch(activation) {
    case RELU:
      relu_activate();
    case LEAKY:
      leaky_activate();
  }
#endif

}

void Relu::relu_activate() {

  for(int i = 0; i < batch; i++) 
    for(int j = 0; j < N; j++) 
      output[i*N+j] = (input[i*N+j] >= 0 ? input[i*N+j] : 0);
}

void Relu::leaky_activate() {

  for(int i = 0; i < batch; i++) 
    for(int j = 0; j < N; j++) 
      output[i*N+j] = (input[i*N+j] >= 0 ? input[i*N+j] : 0.1*input[i*N+j]);
}

void Relu::backward(float *delta) {

#ifdef GPU
  switch(activation) {
    case RELU:
      relu_backward_gpu(delta);
    case LEAKY:
      leaky_backward_gpu(delta);
  }

#else

  switch(activation) {
    case RELU:
      relu_backward(delta);
    case LEAKY:
      leaky_backward(delta);
  }

#endif
}


void Relu::relu_backward(float *delta) {
  for(int i = 0; i < batch; i++) 
    for(int j = 0; j < N; j++) 
      m_delta[i*N+j] = (cut[i*N+j] + delta[i*N+j])*(input[i*N+j] >= 0);
}

void Relu::leaky_backward(float *delta) {
  for(int i = 0; i < batch; i++) 
    for(int j = 0; j < N; j++) 
      m_delta[i*N+j] = (cut[i*N+j] + delta[i*N+j])*(input[i*N+j] >= 0 ? 1.0 : 0.1);
}

void Relu::update(update_args a) {
}

void Relu::save(fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Relu,%d,%d", N, activation);
  //cout << buf << endl;
  file->write(buf, sizeof(buf));
}

Relu* Relu::load(char *buf) {

  int para[2] = {0};
  char *token;
  int idx = 0;
  while (buf) {
    token = strtok(NULL, ",");
    para[idx] = atoi(token);
    idx++;
    if(idx > 2)
      break;
  }

  Relu *relu = new Relu(para[0], (ACT)para[1]);
  return relu;
}
