#include "layer/dropout.h"

__global__ void dropout_forward_gpu_kernel(float *output, float *input, float *mask_, float *prob, int ratio_, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index >= size) return;
  mask_[index] = (prob[index] > ratio_ ? 1.0 : 0.0);
  output[index] = input[index]*mask_[index];
}


__global__ void dropout_copy_kernel(float *output, float *input, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index >= size) return;
  output[index] = input[index];
}


void Dropout::Init() {
  output = malloc_gpu(batch*n_);
  delta_ = malloc_gpu(batch*n_);
  mask_ = malloc_gpu(batch*n_);
  prob_ = malloc_gpu(batch*n_);
}

void Dropout::Forward() {

  if(train_flag_) {
    srand(time(NULL));

    float *prob_tmp = new float[n_*batch];
    for(int i = 0; i < batch; i++) 
      for(int j = 0; j < n_; j++) 
        prob_tmp[i*n_ + j] = (float)rand()/(RAND_MAX + 1.0);

    gpu_push_array(prob_, prob_tmp, n_*batch);
    int size = batch*n_;
    dropout_forward_gpu_kernel<<<default_grid(size), BLOCK>>>(output, input, mask_, prob_, ratio_, size);
    check_error(cudaGetLastError());
    delete []prob_tmp;
  }
  else {
    int size = batch*n_;
    dropout_copy_kernel<<<default_grid(size), BLOCK>>>(output, input, size);
  }


}


__global__ void dropout_backward_gpu_kernel(float *delta_, float *delta, float *mask_, int size) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index >= size) return;
  delta_[index] = delta[index]*mask_[index];
}

void Dropout::Backward(float *delta) {
    int size = batch*n_;
    dropout_backward_gpu_kernel<<<default_grid(size), BLOCK>>>(delta_, delta, mask_, size);
}

