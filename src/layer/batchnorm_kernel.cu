#include "layer/batchnorm.h"
#include "blas.h"

void Batchnorm::Init() {

  mean = malloc_gpu(channel_);
  var  = malloc_gpu(channel_);
  running_mean = malloc_gpu(channel_);
  running_var  = malloc_gpu(channel_);

  std = malloc_gpu(channel_);

  normal = malloc_gpu(batch*spatial_*channel_);
  output = malloc_gpu(batch*spatial_*channel_);


  delta_ = malloc_gpu(batch*spatial_*channel_);
  xc = malloc_gpu(batch*channel_);

  dxn = malloc_gpu(batch*spatial_*channel_);
  dxc = malloc_gpu(batch*spatial_*channel_);
  dvar = malloc_gpu(channel_);
  dstd = malloc_gpu(channel_);
  dmu = malloc_gpu(channel_);

  gamma = malloc_gpu(channel_);
  beta = malloc_gpu(channel_);
  dgamma = malloc_gpu(channel_);
  dbeta = malloc_gpu(channel_);
  m_gamma = malloc_gpu(channel_);
  m_beta = malloc_gpu(channel_);
  v_gamma = malloc_gpu(channel_);
  v_beta = malloc_gpu(channel_);

  memset_gpu(gamma, 1.0, channel_);
}

// See https://github.com/pjreddie/darknet/blob/master/src/blas_kernels.cu
__global__ void  fast_mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + (i + id)*filters + filter;
            local[id] += (i+id < spatial) ? x[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean[filter] += local[i];
        }
        mean[filter] /= spatial * batch;
    }
}

void Batchnorm::GetMean() {

  fast_mean_kernel<<<channel_, BLOCK>>>(input, batch, channel_, spatial_, mean);
  check_error(cudaGetLastError());

}

// See https://github.com/pjreddie/darknet/blob/master/src/blas_kernels.cu
__global__ void  fast_variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + (i + id)*filters + filter;

            local[id] += (i+id < spatial) ? powf((x[index] - mean[filter]), 2) : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance[filter] += local[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}



void Batchnorm::GetVariance() {
  fast_variance_kernel<<<channel_, BLOCK>>>(input, mean, batch, channel_, spatial_, var);
  check_error(cudaGetLastError());

}

__global__ void get_running_variable(float momentum, float *running_x, float *x, int n) {

  int j = blockIdx.x*blockDim.x + threadIdx.x;
  if(j >= n) return;
  running_x[j] = momentum*running_x[j] + (1.0 - momentum)*x[j];

}

__global__ void normalize_gpu_kernel(float *normal, float *input, float *mean,  float *var, float epsilon, int n, int channel) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index >= n)
    return;

  int c = index%channel;
  normal[index] = (input[index] - mean[c])/pow(var[c] + epsilon, 0.5);
}

void Batchnorm::Normalize() {

  if(train_flag_) {

    int n = batch*spatial_*channel_;
    normalize_gpu_kernel<<<default_grid(n), BLOCK>>>(normal, input, mean, var, epsilon, n, channel_);
    check_error(cudaGetLastError());

    get_running_variable<<<default_grid(channel_), BLOCK>>>(
		    momentum, running_mean, mean, channel_);
    check_error(cudaGetLastError());

    get_running_variable<<<default_grid(channel_), BLOCK>>>(
		    momentum, running_var, var, channel_);
    check_error(cudaGetLastError());


  }
  else {
    int n = batch*spatial_*channel_;
    normalize_gpu_kernel<<<default_grid(n), BLOCK>>>(normal, input, mean, var, epsilon, n, channel_);
    check_error(cudaGetLastError());
  }
}

__global__ void ScaleAndShift_gpu_kernel(float *output, float *normal, float *gamma, float *beta) {

  int j = blockIdx.x;
  int index = gridDim.x*threadIdx.x + blockIdx.x;
  output[index] = gamma[j]*normal[index] + beta[j];

}

__global__ void scale_and_shift_gpu_kernel(float *output, float *normal, float *gamma, float *beta, int n, int channel) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index >= n)
    return;

  int c = index%channel;
  output[index] = gamma[c]*normal[index] + beta[c];
}


void Batchnorm::ScaleAndShift() {

  int n = batch*spatial_*channel_;
  scale_and_shift_gpu_kernel<<<default_grid(n), BLOCK>>>(output, normal, gamma, beta, n, channel_);
  check_error(cudaGetLastError());

}

void Batchnorm::Forward() {

  GetMean();
  GetVariance();
  Normalize();
  ScaleAndShift();
}



__global__ void cal_mdelta_kernel(float *delta_, float *dxc, float *dmu, 
		int batch, int spatial, int channel) {


  int index = blockDim.x*blockIdx.x + threadIdx.x;
  if(index >= (batch*spatial*channel)) return;
  int i = index%channel;
  delta_[index] = dxc[index] - dmu[i]/(float)(batch*spatial);

}


__global__ void calc_dxc2(float *dxc, float *input, float *mean, float *dvar,
		int batch, int spatial, int channel) {

  int index = blockDim.x*blockIdx.x + threadIdx.x;
  if(index >= (batch*spatial*channel)) return;
  int i = index%channel;
  float tmp = (float)(batch*spatial);

  dxc[index] += (2.0/tmp)*(input[index] - mean[i])*dvar[i];
}



__global__ void get_variance_delta_kernel(float *dvar, float *var, int channel) {
  
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index >= channel) return;
  float epsilon = 1.0e-7;
  dvar[index] = 0.5*dvar[index]/pow(var[index] + epsilon, 0.5);
}


__global__ void cal_dx(float *dxn, float *dxc, float *gamma, float *delta, float *var, float *input, float *mean, int n, int channel) {

  int index = blockDim.x*blockIdx.x + threadIdx.x;
  if(index >= n) return;
  int i = index%channel;
  float epsilon = 1.0e-7;

  float tmp = gamma[i]*delta[index];
  dxn[index] = -1.0*tmp*(input[index] - mean[i])/(var[i] + epsilon);
  dxc[index] = tmp/pow(var[i] + epsilon, 0.5);
}




void Batchnorm::Backward(float *delta) {
  col_sum_gpu(batch*spatial_, channel_, delta, dbeta);

  elementwise_mul_gpu(normal, delta, normal, batch*spatial_*channel_);
  col_sum_gpu(batch*spatial_, channel_, normal, dgamma);
  int n = batch*spatial_*channel_;

  cal_dx<<<default_grid(n), BLOCK>>>(dxn, dxc, gamma, delta, var, input, mean, n, channel_);
  check_error(cudaGetLastError());
  col_sum_gpu(batch*spatial_, channel_, dxn, dvar);
  get_variance_delta_kernel<<<default_grid(channel_), BLOCK>>>(dvar, var, channel_);

 
  calc_dxc2<<<default_grid(n), BLOCK>>>(dxc, input, mean, dvar, batch, spatial_, channel_);
  check_error(cudaGetLastError());
  
  col_sum_gpu(batch*spatial_, channel_, dxc, dmu);
  
  cal_mdelta_kernel<<<default_grid(n), BLOCK>>>(delta_, dxc, dmu, batch, spatial_, channel_);
  check_error(cudaGetLastError());
}



void Batchnorm::Update(UpdateArgs a) {

  if(a.adam) {
    adam_gpu(channel_, gamma, dgamma, m_gamma, v_gamma, a);
    adam_gpu(channel_, beta, dbeta, m_beta, v_beta, a);
  }
  else {
    momentum_gpu(channel_, gamma, dgamma, v_gamma, a);
    momentum_gpu(channel_, beta, dbeta, v_beta, a);
  }
}

