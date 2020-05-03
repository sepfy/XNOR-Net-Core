#include "layer/dropout.h"

__global__ void dropout_forward_gpu_kernel(float *output, float *input, float *mask, float *prob, int ratio, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index >= size) return;
  mask[index] = (prob[index] > ratio ? 1.0 : 0.0);
  output[index] = input[index]*mask[index];
}


__global__ void dropout_copy_kernel(float *output, float *input, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index >= size) return;
  output[index] = input[index];
}


void Dropout::forward() {


  if(train_flag) {
    srand(time(NULL));

    float *prob_tmp = new float[N*batch];
    for(int i = 0; i < batch; i++) 
      for(int j = 0; j < N; j++) 
        prob_tmp[i*N + j] = (float)rand()/(RAND_MAX + 1.0);

    gpu_push_array(prob, prob_tmp, N*batch);
    int size = batch*N;
    dropout_forward_gpu_kernel<<<default_grid(size), BLOCK>>>(output, input, mask, prob, ratio, size);
    check_error(cudaGetLastError());
    delete []prob_tmp;
  }
  else {
    int size = batch*N;
    dropout_copy_kernel<<<default_grid(size), BLOCK>>>(output, input, size);
  }


}


__global__ void dropout_backward_gpu_kernel(float *m_delta, float *delta, float *mask, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index >= size) return;
  m_delta[index] = delta[index]*mask[index];
}

void Dropout::backward(float *delta) {
    int size = batch*N;
    dropout_backward_gpu_kernel<<<default_grid(size), BLOCK>>>(m_delta, delta, mask, size);
}

