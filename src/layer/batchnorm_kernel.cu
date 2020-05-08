#include "layer/batchnorm.h"
#include "blas.h"

void Batchnorm::Init() {

  mean = malloc_gpu(n_);

  std = malloc_gpu(n_);
  var  = malloc_gpu(n_);

  running_mean = malloc_gpu(n_);
  running_var  = malloc_gpu(n_);
  normal = malloc_gpu(batch*n_);
  output = malloc_gpu(batch*n_);
  delta_ = malloc_gpu(batch*n_);

  xc = malloc_gpu(batch*n_);
  dxn = malloc_gpu(batch*n_);
  dxc = malloc_gpu(batch*n_);
  dvar = malloc_gpu(n_);
  dstd = malloc_gpu(n_);
  dmu = malloc_gpu(n_);

  dgamma = malloc_gpu(n_);
  dbeta = malloc_gpu(n_);
  gamma = malloc_gpu(n_);
  beta = malloc_gpu(n_);
  m_gamma = malloc_gpu(n_);
  m_beta = malloc_gpu(n_);
  v_gamma = malloc_gpu(n_);
  v_beta = malloc_gpu(n_);

  memset_gpu(gamma, 1.0, n_);
}


__global__ void mean_gpu_kernel(float *input, float *mean, float batch, int n_) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index >= n_)
    return;
  
  mean[index] = 0.0;
  int i;
  for(i = 0; i < batch; i++)
    mean[index] += input[i*n_ + index];

  mean[index] /= batch;

}


void Batchnorm::GetMean() {

  mean_gpu_kernel<<<default_grid(n_),BLOCK>>>(input, mean, batch, n_);  
  check_error(cudaGetLastError());

}

/*
__global__ void calc_xc(float *input, float *mean, float *xc) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.x; 
}
*/

__global__ void variance_gpu_kernel(float *input, float *mean, float *var, float batch, int n_) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index >= n_)
    return;
  
  var[index] = 0.0;
  int i;

  //xc[i] = (pow(input[i] - mean[j], 2.0));
  for(i = 0; i < batch; i++)
    var[index] += pow(input[i*n_ + index] - mean[index], 2.0);

  var[index] /= batch;


}

void Batchnorm::GetVariance() {

  variance_gpu_kernel<<<default_grid(n_),BLOCK>>>(input, mean, var, batch, n_);  
  check_error(cudaGetLastError());

}

__global__ void Normalize_gpu_kernel(float *normal, float *input, float *mean,  float *var, float epsilon) {

  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  normal[index] = (input[index] - mean[j])/pow(var[j] + epsilon, 0.5);

}

__global__ void get_running_variable(float momentum, float *running_x, float *x, int n) {

  int j = blockIdx.x*blockDim.x + threadIdx.x;
  if(j >= n) return;
  running_x[j] = momentum*running_x[j] + (1.0 - momentum)*x[j];

}

void Batchnorm::Normalize() {

  if(train_flag_) {

    Normalize_gpu_kernel<<<n_, batch>>>(normal, input, mean, var, epsilon);
    check_error(cudaGetLastError());

    get_running_variable<<<default_grid(n_), BLOCK>>>(
		    momentum, running_mean, mean, n_);
    check_error(cudaGetLastError());

    get_running_variable<<<default_grid(n_), BLOCK>>>(
		    momentum, running_var, var, n_);
    check_error(cudaGetLastError());
  }
  else {
    Normalize_gpu_kernel<<<n_, batch>>>(
		    normal, input, running_mean, running_var, epsilon);
    check_error(cudaGetLastError());
  }
}

__global__ void ScaleAndShift_gpu_kernel(float *output, float *normal, float *gamma, float *beta) {

  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  output[index] = gamma[j]*normal[index] + beta[j];

}


void Batchnorm::ScaleAndShift() {

  ScaleAndShift_gpu_kernel<<<n_, batch>>>(output, normal, gamma, beta);
  check_error(cudaGetLastError());

}

void Batchnorm::Forward() {

  GetMean();
  GetVariance();
  Normalize();
  ScaleAndShift();
}


__global__ void cal_dx(float *dxn, float *dxc, float *gamma, float *delta, float *var, float *xc, float *input, float *mean, float epsilon) {

  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  
  dxn[index] = gamma[j]*delta[index];
  dxc[index] = dxn[index]/pow(var[j] + epsilon, 0.5);
  dxn[index] = -1.0*dxn[index]*(input[index] - mean[j])/(var[j] + epsilon);
  //dxn[index] = -1.0*(dxn[index]*(pow(input[index] - mean[j], 2.0)))/(var[j] + epsilon);
}

__global__ void cal_dvar_kernel(float *dvar, float *dstd, float *var, float epsilon, int n_) {
  
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  if(j >= n_) return;
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

  col_sum_gpu(batch, n_, delta, dbeta);

  elementwise_mul_gpu(normal, delta, normal, batch*n_);

  col_sum_gpu(batch, n_, normal, dgamma);

  cal_dx<<<n_, batch>>>(dxn, dxc, gamma, delta, var, xc, input, mean, epsilon);
  check_error(cudaGetLastError());

  col_sum_gpu(batch, n_, dxn, dstd);

  cal_dvar_kernel<<<default_grid(n_),BLOCK>>>(dvar, dstd, var, epsilon, n_);
  check_error(cudaGetLastError());

  calc_dxc2<<<n_, batch>>>(dxc, input, mean, dvar, (float)batch);
  check_error(cudaGetLastError());

  col_sum_gpu(batch, n_, dxc, dmu);

  cal_mdelta_kernel<<<n_, batch>>>(delta_, dxc, dmu, (float)batch);
  check_error(cudaGetLastError());

}

void Batchnorm::Update(UpdateArgs a) {

  if(a.adam) {
    adam_gpu(n_, gamma, dgamma, m_gamma, v_gamma, a);
    adam_gpu(n_, beta, dbeta, m_beta, v_beta, a);
  }
  else {
    momentum_gpu(n_, gamma, dgamma, v_gamma, a);
    momentum_gpu(n_, beta, dbeta, v_beta, a);
  }
}

