#include "layers.h"

Connected::Connected(int _n, int _m) {
  N = _n;
  M = _m;
}

Connected::~Connected() {

}

void Connected::init() {

  weight = new float[N*M];
  bias   = new float[M];
  output = new float[batch*M];

  grad_weight = new float[N*M];
  grad_bias = new float[M];
  m_delta = new float[batch*N];

  random_normal(N*M, weight);
  random_normal(M, bias);
}

void Connected::forward() {  
  gemm(batch, M, N, 1, input, weight, output);
  bias_add(batch, M, output, bias);
}

void Connected::backward(float *delta) {

  memset(grad_weight, 0, N*M*sizeof(float));
  memset(m_delta, 0, batch*N*sizeof(float));

  gemm_ta(N, M, batch, 1.0, input, delta, grad_weight);
  gemm_tb(batch, N, M, 1.0, delta, weight, m_delta);
  row_sum(batch, M, delta, grad_bias);
}

void Connected::update(float lr) {
  mat_scalar(N, M, grad_weight, lr, grad_weight);
  mat_minus(N, M, weight, grad_weight, weight);
  mat_scalar(1, M, grad_bias, lr, grad_bias);
  mat_minus(1, M, bias, grad_bias, bias);
}

