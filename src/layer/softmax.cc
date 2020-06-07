#include "softmax.h"

void SoftmaxWithCrossEntropy::Print() {
  printf("Softmax\n");
}

#ifndef GPU

void SoftmaxWithCrossEntropy::Init() {
  output = new float[batch*n_];
  delta_ = new float[batch*n_];
}

void SoftmaxWithCrossEntropy::Forward() {

  for(int i = 0; i < batch; i++) {
    float tmp = 0;
    float max = 0;
    for(int j = 0; j < n_; j++) 
      if(input[i*n_+j] > max)
	max = input[i*n_+j];

    for(int j = 0; j < n_; j++) {
      output[i*n_+j] = exp(input[i*n_+j] - max);
      tmp += output[i*n_+j];
    }
    for(int j = 0; j < n_; j++) 
      output[i*n_+j] /= tmp;
  }

}

void SoftmaxWithCrossEntropy::Backward(float *delta) {

  mat_minus(batch, n_, output, delta, delta_);  
  mat_scalar(batch, n_, delta_, 1.0/(float)batch, delta_);
}
#endif


void SoftmaxWithCrossEntropy::Save(std::fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Softmax,%d", n_);
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

void SoftmaxWithCrossEntropy::LoadParams(std::fstream *file, int batch) {}
