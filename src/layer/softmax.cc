#include "softmax.h"

SoftmaxWithCrossEntropy::SoftmaxWithCrossEntropy(int n) {
  N = n;
}

void SoftmaxWithCrossEntropy::Init() {

#ifdef GPU
  output = malloc_gpu(batch*N);
  delta_ = malloc_gpu(batch*N);
#else
  output = new float[batch*N];
  delta_ = new float[batch*N];
#endif

}

SoftmaxWithCrossEntropy::~SoftmaxWithCrossEntropy() {

#ifdef GPU
#else
  delete []output;
  delete []delta_;
#endif

}    

void SoftmaxWithCrossEntropy::Print() {

  float umem = (float)(2*batch*N)/(1024*1024);
  printf("Softmax\t %.2f\n", umem);
}

#ifndef GPU
void SoftmaxWithCrossEntropy::Forward() {

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

void SoftmaxWithCrossEntropy::Backward(float *delta) {

  mat_minus(batch, N, output, delta, delta_);  
  mat_scalar(batch, N, delta_, 1.0/(float)batch, delta_);
}
#endif


void SoftmaxWithCrossEntropy::Save(std::fstream *file) {
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


