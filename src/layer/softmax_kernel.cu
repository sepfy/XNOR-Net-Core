#include "softmax.h"

__global__ void softmax_kernel(float *input, float *output, int n_) {

    int i = threadIdx.x;
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

void SoftmaxWithCrossEntropy::Init() {
  output = malloc_gpu(batch*n_);
  delta_ = malloc_gpu(batch*n_);
}

void SoftmaxWithCrossEntropy::Forward() {

  softmax_kernel<<<1, batch>>>(input, output, n_);
  check_error(cudaGetLastError());
}

void SoftmaxWithCrossEntropy::Backward(float *delta) {
  float alpha = 1.0/(float)batch;
  size_t size = sizeof(float)*batch*n_;
  cudaError_t status = cudaMemset(delta_, 0, size);
  check_error(status);

  cublasSaxpy(gpu_handle(), batch*n_, &alpha, output, 1, delta_, 1);
  check_error(cudaGetLastError());

  alpha = -1.0/(float)batch;
  cublasSaxpy(gpu_handle(), batch*n_, &alpha, delta, 1, delta_, 1);
  check_error(cudaGetLastError());
}


