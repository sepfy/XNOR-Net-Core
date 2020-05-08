#include "layer/connected.h"

void Connected::Init() {

  output = malloc_gpu(batch*m_);

  weight = malloc_gpu(n_*m_);
  bias   = malloc_gpu(m_);
  grad_weight = malloc_gpu(n_*m_);
  grad_bias = malloc_gpu(m_);

  delta_ = malloc_gpu(batch*n_);

  // Adam optimizer
  m_weight = malloc_gpu(n_*m_);
  v_weight = malloc_gpu(n_*m_);
  m_bias = malloc_gpu(m_);
  v_bias = malloc_gpu(m_);

  random_normal_gpu(n_*m_, weight);
  random_normal_gpu(m_, bias);

}

void Connected::Forward() {
  gemm_gpu(TRS_N, TRS_N, batch, m_, n_, 1, input, weight, output);
  bias_add_gpu(output, bias, batch, 1, m_);
}

void Connected::Backward(float *delta) {

  gemm_gpu(TRS_N, TRS_T, batch, n_, m_, 1.0, delta, weight, delta_);
  gemm_gpu(TRS_T, TRS_N, n_, m_, batch, 1.0, input, delta, grad_weight);
  row_sum_gpu(batch, m_, delta, grad_bias);
}

void Connected::Update(UpdateArgs update_args) {
  axpy_gpu(n_*m_, update_args.decay, weight, grad_weight);
  //axpy_gpu(m_, a.decay, bias, grad_bias);

  if(update_args.adam) {
    adam_gpu(n_*m_, weight, grad_weight, m_weight, v_weight, update_args);
    adam_gpu(m_, bias, grad_bias, m_bias, v_bias, update_args);
  }
  else {
    momentum_gpu(n_*m_, weight, grad_weight, v_weight, update_args);
    momentum_gpu(m_, bias, grad_bias, v_bias, update_args);
  }
}
