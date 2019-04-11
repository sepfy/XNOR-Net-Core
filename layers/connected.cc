#include "layers.h"

Connected::Connected(int _batch, int _n, int _m, float *_input) {
  batch = _batch;
  N = _n;
  M = _m;
  input = _input;
  init();
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

  srand(time(NULL));
  for(int j = 0; j < M; j++) {
    bias[j] = 0.1*((float) rand()/(RAND_MAX + 1.0) - 0.5);
    for(int i = 0; i < N; i++) {
      weight[i*M+j] = 0.1*((float) rand()/(RAND_MAX + 1.0) -0.5);
    }
  }
}

void Connected::forward() {  

  memset(output, 0, batch*M*sizeof(float));
  bias_add(batch, M, output, bias);
  gemm(batch, M, N, 1, input, weight, output);
}

void Connected::backward(float *delta) {

  memset(grad_weight, 0, N*M*sizeof(float));
  memset(m_delta, 0, batch*N*sizeof(float));

  gemm_ta(N, M, batch, 1.0, input, delta, grad_weight);
  gemm_tb(batch, N, M, 1.0, delta, weight, m_delta);
  row_sum(batch, M, delta, grad_bias);
}

void Connected::update() {
  mat_scalar(N, M, grad_weight, 0.1, grad_weight);
  mat_minus(N, M, weight, grad_weight, weight);
  mat_scalar(1, M, grad_bias, 0.1, grad_bias);
  mat_minus(1, M, bias, grad_bias, bias);
}

