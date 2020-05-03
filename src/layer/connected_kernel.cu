#include "layer/connected.h"

void Connected::init() {

  output = malloc_gpu(batch*M);

  weight = malloc_gpu(N*M);
  bias   = malloc_gpu(M);
  grad_weight = malloc_gpu(N*M);
  grad_bias = malloc_gpu(M);

  m_delta = malloc_gpu(batch*N);

  // Adam optimizer
  m_weight = malloc_gpu(N*M);
  v_weight = malloc_gpu(N*M);
  m_bias = malloc_gpu(M);
  v_bias = malloc_gpu(M);

  random_normal_gpu(N*M, weight);
  random_normal_gpu(M, bias);

}


void Connected::forward() {
  gemm_gpu(TRS_N, TRS_N, batch, M, N, 1, input, weight, output);
  bias_add_gpu(output, bias, batch, 1, M);
}

void Connected::backward(float *delta) {

  gemm_gpu(TRS_N, TRS_T, batch, N, M, 1.0, delta, weight, m_delta);
  gemm_gpu(TRS_T, TRS_N, N, M, batch, 1.0, input, delta, grad_weight);
  row_sum_gpu(batch, M, delta, grad_bias);
}

void Connected::update(update_args a) {
  axpy_gpu(N*M, a.decay, weight, grad_weight);
  //axpy_gpu(M, a.decay, bias, grad_bias);

  if(a.adam) {
    adam_gpu(N*M, weight, grad_weight, m_weight, v_weight, a);
    adam_gpu(M, bias, grad_bias, m_bias, v_bias, a);
  }
  else {
    momentum_gpu(N*M, weight, grad_weight, v_weight, a);
    momentum_gpu(M, bias, grad_bias, v_bias, a);
  }
}
