#include "layers.h"
#include "gpu.h"


void Batchnorm::get_mean_gpu() {

  cudaMemset(mean, 0, sizeof(float)*N);
  check_error(cudaGetLastError());
  float alpha = 1/(float)batch;
  for(int i = 0; i < batch; i++)
    cublasSaxpy(gpu_handle(), N, &alpha, input + i*N, 1, mean, 1);
  check_error(cudaGetLastError());

}

__global__ void calc_xc(float *input, float *mean, float *xc) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.x; 
  xc[i] = (pow(input[i] - mean[j], 2.0));
}

void Batchnorm::get_variance_gpu() {

  cudaMemset(var, 0, sizeof(float)*N);
  check_error(cudaGetLastError());
  calc_xc<<<N, batch>>>(input, mean, xc);
  float alpha = 1/(float)batch;
  for(int i = 0; i < batch; i++)
    cublasSaxpy(gpu_handle(), N, &alpha, xc + i*N, 1, var, 1);
  check_error(cudaGetLastError());
  //cudaDeviceSynchronize();

}

__global__ void normalize_gpu_kernel(float *normal, float *input, float *mean,  float *var, float epsilon) {

  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  normal[index] = (input[index] - mean[j])/pow(var[j] + epsilon, 0.5);

}

__global__ void get_running_variable(float momentum, float *running_x, float *x) {

  int j = blockIdx.x*blockDim.x + threadIdx.x;
  running_x[j] = momentum*running_x[j] + (1.0 - momentum)*x[j];

}

void Batchnorm::normalize_gpu() {

  int grid = (N-1)/256 + 1;
  normalize_gpu_kernel<<<N, batch>>>(normal, input, mean, var, epsilon);
  get_running_variable<<<grid, 256>>>(momentum, running_mean, mean);
  check_error(cudaGetLastError());
  get_running_variable<<<grid, 256>>>(momentum, running_var, var);
  check_error(cudaGetLastError());
//  cudaDeviceSynchronize();

}

__global__ void scale_and_shift_gpu_kernel(float *output, float *normal, float *gamma, float *beta) {


  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  output[index] = gamma[j]*normal[index] + beta[j];

}



void Batchnorm::scale_and_shift_gpu() {

  scale_and_shift_gpu_kernel<<<N, batch>>>(output, normal, gamma, beta);
  check_error(cudaGetLastError());
//  cudaDeviceSynchronize();

}


__global__ void ew_mul_kernel(float *A, float *B, float *C) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  C[i] = A[i] + B[i];
}


__global__ void cal_dx(float *dxn, float *dxc, float *gamma, float *delta, float *var, float *xc, float *input, float epsilon) {

  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  
  dxn[index] = gamma[j]*delta[index];
  dxc[index] = dxn[index]/pow(var[j] + epsilon, 0.5);
  //dxn[index] = -1.0*dxn[index]*(input[index] - mean[j])/(var[j] + epsilon);
  dxn[index] *= -1.0*(xc[index])/(var[j] + epsilon);
}

__global__ void cal_dvar_kernel(float *dvar, float *dstd, float *var, float epsilon) {
  
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  dvar[j] = 0.5*dstd[j]/pow(var[j] + epsilon, 0.5);
}





__global__ void cal_mdelta_kernel(float *m_delta, float *dxc, float *dmu, float batch) {


  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  m_delta[index] = dxc[index] - dmu[j]/batch;

}


void col_sum_gpu3(int N, int M, float *A, float *B) {

  //memset(B, 0, M*sizeof(float));
  cudaMemset(B, 0, sizeof(float)*M);
  check_error(cudaGetLastError());
  float alpha = 1.0;
  float beta = 0.0;
  float *e = malloc_gpu(N);
  cudaMemset(e, 1.0, sizeof(float)*N);
  check_error(cudaGetLastError());
  cublasSgemv(gpu_handle(), CUBLAS_OP_T, M, N, &alpha, A, M, e, 1, &beta, B, 1);
  check_error(cudaGetLastError());
  cudaFree(e);
}

__global__ void calc_dxc2(float *dxc, float *xc, float *dvar, float batch) {

  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  dxc[index] += (2.0/batch)*(xc[index])*dvar[j];
}

void Batchnorm::backward_gpu(float *delta) {

  col_sum_gpu3(batch, N, delta, dbeta);

  float *tmp = malloc_gpu(batch*N);
  ew_mul_kernel<<<N, batch>>>(normal, delta, tmp);
  col_sum_gpu3(batch, N, tmp, dgamma);

  cal_dx<<<N, batch>>>(dxn, dxc, gamma, delta, var, xc, input, epsilon);
  col_sum_gpu3(batch, N, dxn, dstd);

  int grid = (N - 1)/256 + 1;
  cal_dvar_kernel<<<grid, 256>>>(dvar, dstd, var, epsilon);

  calc_dxc2<<<N, batch>>>(dxc, xc, dvar, (float)batch);

  col_sum_gpu3(batch, N, dxc, dmu);
  cal_mdelta_kernel<<<N, batch>>>(m_delta, dxc, dmu, (float)batch);

  cudaFree(tmp);
  check_error(cudaGetLastError());
  //cudaDeviceSynchronize();
}


