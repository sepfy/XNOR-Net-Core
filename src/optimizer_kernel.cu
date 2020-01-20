#include "layers.h"

#ifdef GPU
#include "gpu.h"
#endif


__global__ void adam_gpu_kernel(int n, float *x, float *grad_x, float *m_x, float *v_x, 
		float beta1, float beta2, float m_lr, float epsilon) {

  int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;

  m_x[i] = (1 - beta1)*grad_x[i] + beta1*m_x[i];
  v_x[i] = (1 - beta2)*pow(grad_x[i], 2.0) + beta2*v_x[i];
  x[i] -= m_lr * m_x[i]/(pow(v_x[i], 0.5) + epsilon);

}

void adam_gpu(int n, float *x, float *grad_x, float *m_x, float *v_x, update_args a) {

  float m_lr = a.lr * pow(1.0 - pow(a.beta2, a.iter), 0.5) / (1.0 - pow(a.beta1, a.iter));
  unsigned int grid = n/512+1;
  adam_gpu_kernel<<<grid, 512>>>(n, x, grad_x, m_x, v_x, a.beta1, a.beta2, m_lr, a.epsilon);
  cudaDeviceSynchronize();

}


