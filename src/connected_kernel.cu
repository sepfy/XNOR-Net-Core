#include "layers.h"

void Connected::forward_gpu() {
  gemm_gpu(TRS_N, TRS_N, batch, M, N, 1, input, weight, output);
  bias_add_gpu(output, bias, batch, 1, M);
}

void Connected::backward_gpu(float *delta) {

  gemm_gpu(TRS_N, TRS_T, batch, N, M, 1.0, delta, weight, m_delta);
  gemm_gpu(TRS_T, TRS_N, N, M, batch, 1.0, input, delta, grad_weight);
  row_sum_gpu(batch, M, delta, grad_bias);
}
