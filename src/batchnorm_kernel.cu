#include "layers.h"
#include "gpu.h"


void Batchnorm::get_mean_gpu() {

  memset(mean, 0, N*sizeof(float));

  float alpha = 1/(float)batch;
  for(int i = 0; i < batch; i++)
    cublasSaxpy(gpu_handle(), N, &alpha, input + i*N, 1, mean, 1);
  cudaDeviceSynchronize();

}


__global__ void get_variance_gpu_kernel(float *input, float *mean, float *var, float batch) {

  int j = blockIdx.x*blockDim.x + threadIdx.x;
  var[j] += (pow(input[j] - mean[j], 2.0))/(float)batch;
}



void Batchnorm::get_variance_gpu() {

  memset(var, 0, N*sizeof(float));
  for(int i = 0; i < batch; i++) {
    int grid = (N-1)/256 + 1;
    get_variance_gpu_kernel<<<grid, 256>>>(input + i*N, mean, var, (float)batch);
  }
  cudaDeviceSynchronize();

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
  get_running_variable<<<grid, 256>>>(momentum, running_var, var);
  cudaDeviceSynchronize();

}

__global__ void scale_and_shift_gpu_kernel(float *output, float *normal, float *gamma, float *beta) {


  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  output[index] = gamma[j]*normal[index] + beta[j];

}



void Batchnorm::scale_and_shift_gpu() {

  scale_and_shift_gpu_kernel<<<N, batch>>>(output, normal, gamma, beta);
  cudaDeviceSynchronize();

}

/*
__global__ void backward_gpu_kernel(float *delta, float *gamma, float *dxn, float *dxc, float *var, float epsilon) {

  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;


  dxn[index] = gamma[j]*delta[index];
  dxc[index] = dxn[index]/pow(var[j] + epsilon, 0.5);
}
*/

__global__ void ew_mul_kernel(float *A, float *B, float *C) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  C[i] = A[i] + B[i];
}


__global__ void cal_dx(float *dxn, float *dxc, float *gamma, float *delta, float *var, float *mean, float *input, float epsilon) {

  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  
  dxn[index] = gamma[j]*delta[index];
  dxc[index] = dxn[index]/pow(var[j] + epsilon, 0.5);
  dxn[index] = dxn[index]*(input[index] - mean[j])/(var[j] + epsilon);

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



__global__ void col_sum_gpu_kernel(int N, float alpha, float *A, float *B) {

  int j = threadIdx.x;
  *B += *(A + j*N)*alpha;
}


void col_sum_gpu(int batch, int N, float alpha, float *A, float *B) {

  memset(B, 0, sizeof(float)*N);
  for(int i = 0; i < N; i++)
    col_sum_gpu_kernel<<<1, batch>>>(N, alpha, A+i, B+i);

}


void Batchnorm::backward_gpu(float *delta) {


  memset(dbeta, 0 , N*sizeof(float));
  memset(dgamma, 0 , N*sizeof(float));

  col_sum(batch, N, delta, dbeta);

  float *tmp = malloc_gpu(batch*N);
  ew_mul_kernel<<<N, batch>>>(normal, delta, tmp);
  col_sum(batch, N, tmp, dgamma);
  cudaDeviceSynchronize();



  cal_dx<<<N, batch>>>(dxn, dxc, gamma, delta, var, mean, input, epsilon);

  col_sum_gpu(batch, N, -1.0, dxn, dstd);


  int grid = (N - 1)/256 + 1;
  cal_dvar_kernel<<<grid, 256>>>(dvar, dstd, var, epsilon);

  // Step3+4
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      dxc[i*N+j] += (2.0/(float)batch)*(input[i*N+j] - mean[j])*dvar[j];
    }
  }


  // Step1
  memset(dmu, 0, N*sizeof(float));
  col_sum_gpu(batch, N, 1.0, dxc, dmu);


  cal_mdelta_kernel<<<N, batch>>>(m_delta, dxc, dmu, (float)batch);

  cudaFree(tmp);
}


