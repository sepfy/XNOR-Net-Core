#include "layer/batchnorm.h"
#include "blas.h"

void Batchnorm::Init() {

  mean = malloc_gpu(N);

  std = malloc_gpu(N);
  var  = malloc_gpu(N);

  running_mean = malloc_gpu(N);
  running_var  = malloc_gpu(N);
  normal = malloc_gpu(batch*N);
  output = malloc_gpu(batch*N);
  delta_ = malloc_gpu(batch*N);

  xc = malloc_gpu(batch*N);
  dxn = malloc_gpu(batch*N);
  dxc = malloc_gpu(batch*N);
  dvar = malloc_gpu(N);
  dstd = malloc_gpu(N);
  dmu = malloc_gpu(N);

  dgamma = malloc_gpu(N);
  dbeta = malloc_gpu(N);
  gamma = malloc_gpu(N);
  beta = malloc_gpu(N);
  m_gamma = malloc_gpu(N);
  m_beta = malloc_gpu(N);
  v_gamma = malloc_gpu(N);
  v_beta = malloc_gpu(N);

  memset_gpu(gamma, 1.0, N);
}


__global__ void mean_gpu_kernel(float *input, float *mean, float batch, int N) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index >= N)
    return;
  
  mean[index] = 0.0;
  int i;
  for(i = 0; i < batch; i++)
    mean[index] += input[i*N + index];

  mean[index] /= batch;

}


void Batchnorm::get_mean_gpu() {

  mean_gpu_kernel<<<default_grid(N),BLOCK>>>(input, mean, batch, N);  
  check_error(cudaGetLastError());

}

/*
__global__ void calc_xc(float *input, float *mean, float *xc) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.x; 
}
*/

__global__ void variance_gpu_kernel(float *input, float *mean, float *var, float batch, int N) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index >= N)
    return;
  
  var[index] = 0.0;
  int i;

  //xc[i] = (pow(input[i] - mean[j], 2.0));
  for(i = 0; i < batch; i++)
    var[index] += pow(input[i*N + index] - mean[index], 2.0);

  var[index] /= batch;


}

void Batchnorm::get_variance_gpu() {

  variance_gpu_kernel<<<default_grid(N),BLOCK>>>(input, mean, var, batch, N);  
  check_error(cudaGetLastError());

}

__global__ void normalize_gpu_kernel(float *normal, float *input, float *mean,  float *var, float epsilon) {

  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  normal[index] = (input[index] - mean[j])/pow(var[j] + epsilon, 0.5);

}

__global__ void get_running_variable(float momentum, float *running_x, float *x, int n) {

  int j = blockIdx.x*blockDim.x + threadIdx.x;
  if(j >= n) return;
  running_x[j] = momentum*running_x[j] + (1.0 - momentum)*x[j];

}

void Batchnorm::normalize_gpu() {

  if(train_flag_) {

    normalize_gpu_kernel<<<N, batch>>>(normal, input, mean, var, epsilon);
    check_error(cudaGetLastError());

    get_running_variable<<<default_grid(N), BLOCK>>>(
		    momentum, running_mean, mean, N);
    check_error(cudaGetLastError());

    get_running_variable<<<default_grid(N), BLOCK>>>(
		    momentum, running_var, var, N);
    check_error(cudaGetLastError());
  }
  else {
    normalize_gpu_kernel<<<N, batch>>>(
		    normal, input, running_mean, running_var, epsilon);
    check_error(cudaGetLastError());
  }
}

__global__ void scale_and_shift_gpu_kernel(float *output, float *normal, float *gamma, float *beta) {

  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  output[index] = gamma[j]*normal[index] + beta[j];

}


void Batchnorm::scale_and_shift_gpu() {

  scale_and_shift_gpu_kernel<<<N, batch>>>(output, normal, gamma, beta);
  check_error(cudaGetLastError());

}

void Batchnorm::Forward() {

  get_mean_gpu();
  get_variance_gpu();
  normalize_gpu();
  scale_and_shift_gpu();
}


__global__ void cal_dx(float *dxn, float *dxc, float *gamma, float *delta, float *var, float *xc, float *input, float *mean, float epsilon) {

  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  
  dxn[index] = gamma[j]*delta[index];
  dxc[index] = dxn[index]/pow(var[j] + epsilon, 0.5);
  dxn[index] = -1.0*dxn[index]*(input[index] - mean[j])/(var[j] + epsilon);
  //dxn[index] = -1.0*(dxn[index]*(pow(input[index] - mean[j], 2.0)))/(var[j] + epsilon);
}

__global__ void cal_dvar_kernel(float *dvar, float *dstd, float *var, float epsilon, int N) {
  
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  if(j >= N) return;
  dvar[j] = 0.5*dstd[j]/pow(var[j] + epsilon, 0.5);
}


__global__ void cal_mdelta_kernel(float *delta_, float *dxc, float *dmu, float batch) {


  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  delta_[index] = dxc[index] - dmu[j]/batch;

}


__global__ void calc_dxc2(float *dxc, float *input, float *mean, float *dvar, float batch) {

  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  dxc[index] += (2.0/batch)*(input[index] - mean[j])*dvar[j];
}

void Batchnorm::Backward(float *delta) {

  col_sum_gpu(batch, N, delta, dbeta);

  elementwise_mul_gpu(normal, delta, normal, batch*N);

  col_sum_gpu(batch, N, normal, dgamma);

  cal_dx<<<N, batch>>>(dxn, dxc, gamma, delta, var, xc, input, mean, epsilon);
  check_error(cudaGetLastError());

  col_sum_gpu(batch, N, dxn, dstd);

  cal_dvar_kernel<<<default_grid(N),BLOCK>>>(dvar, dstd, var, epsilon, N);
  check_error(cudaGetLastError());

  calc_dxc2<<<N, batch>>>(dxc, input, mean, dvar, (float)batch);
  check_error(cudaGetLastError());

  col_sum_gpu(batch, N, dxc, dmu);

  cal_mdelta_kernel<<<N, batch>>>(delta_, dxc, dmu, (float)batch);
  check_error(cudaGetLastError());

}

void Batchnorm::Update(UpdateArgs a) {

  if(a.adam) {
    adam_gpu(N, gamma, dgamma, m_gamma, v_gamma, a);
    adam_gpu(N, beta, dbeta, m_beta, v_beta, a);
  }
  else {
    momentum_gpu(N, gamma, dgamma, v_gamma, a);
    momentum_gpu(N, beta, dbeta, v_beta, a);
  }
}

