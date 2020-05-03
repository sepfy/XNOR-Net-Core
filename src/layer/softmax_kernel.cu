#include "softmax.h"

__global__ void softmax_kernel(float *input, float *output, int N) {

    int i = threadIdx.x;
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

void SoftmaxWithCrossEntropy::forward() {

  softmax_kernel<<<1, batch>>>(input, output, N);
  check_error(cudaGetLastError());
}

void SoftmaxWithCrossEntropy::backward(float *delta) {
  float alpha = 1.0/(float)batch;
  size_t size = sizeof(float)*batch*N;
  cudaError_t status = cudaMemset(m_delta, 0, size);
  check_error(status);

  cublasSaxpy(gpu_handle(), batch*N, &alpha, output, 1, m_delta, 1);
  check_error(cudaGetLastError());

  alpha = -1.0/(float)batch;
  cublasSaxpy(gpu_handle(), batch*N, &alpha, delta, 1, m_delta, 1);
  check_error(cudaGetLastError());
}


