#include "layers.h"

#ifdef GPU
#include "gpu.h"
#endif

Connected::Connected(int _n, int _m) {
  N = _n;
  M = _m;
}

Connected::~Connected() {

}

void Connected::init() {

#ifdef GPU
  weight = malloc_gpu(N*M);
  bias   = malloc_gpu(M);
  output = malloc_gpu(batch*M);
  grad_weight = malloc_gpu(N*M);
  grad_bias = malloc_gpu(M);
  m_delta = malloc_gpu(batch*N);
  // Adam optimizer
  m_weight = malloc_gpu(N*M);
  v_weight = malloc_gpu(N*M);
  m_bias = malloc_gpu(M);
  v_bias = malloc_gpu(M);
#else
  weight = new float[N*M];
  bias   = new float[M];
  output = new float[batch*M];
  grad_weight = new float[N*M];
  grad_bias = new float[M];
  m_delta = new float[batch*N];
  // Adam optimizer 
  m_weight = new float[N*M];
  v_weight = new float[N*M];
  m_bias = new float[M];
  v_bias = new float[M];
#endif

  random_normal(N*M, weight);
  random_normal(M, bias);

  memset(m_weight, 0, N*M*sizeof(float));
  memset(v_weight, 0, N*M*sizeof(float));
  memset(m_bias, 0, M*sizeof(float));
  memset(v_bias, 0, M*sizeof(float));

}

void Connected::bias_add() {

  for(int i = 0; i < batch; i++)
    for(int j  = 0; j < M; j++)
      output[i*M+j] += bias[j];
}

void Connected::forward() {  

#ifdef GPU
  gemm_gpu(TRS_N, TRS_N, batch, M, N, 1, input, weight, output);
#else
  gemm_cpu(TRS_N, TRS_N, batch, M, N, 1, input, weight, output);
#endif

  bias_add();
}

void Connected::backward(float *delta) {

#ifdef GPU
  gemm_gpu(TRS_N, TRS_T, batch, N, M, 1.0, delta, weight, m_delta);
  gemm_gpu(TRS_T, TRS_N, N, M, batch, 1.0, input, delta, grad_weight);
#else
  gemm_cpu(TRS_N, TRS_T, batch, N, M, 1.0, delta, weight, m_delta);
  gemm_cpu(TRS_T, TRS_N, N, M, batch, 1.0, input, delta, grad_weight);
#endif

  row_sum(batch, M, delta, grad_bias);

}

void Connected::update(update_args a) {

#if GPU
  adam_gpu(N*M, weight, grad_weight, m_weight, v_weight, a);
  adam_gpu(M, bias, grad_bias, m_bias, v_bias, a);
#else
  adam_cpu(N*M, weight, grad_weight, m_weight, v_weight, a);
  adam_cpu(M, bias, grad_bias, m_bias, v_bias, a);
#endif  

#if 0
  mat_scalar(N, M, grad_weight, lr, grad_weight);
  mat_minus(N, M, weight, grad_weight, weight);
  mat_scalar(1, M, grad_bias, lr, grad_bias);
  mat_minus(1, M, bias, grad_bias, bias);
#endif
}

void Connected::save(fstream *file) {

  char buf[64] = {0};
  sprintf(buf, "Connected,%d,%d", N, M);
  //cout << buf << endl;
  file->write(buf, sizeof(buf));
  file->write((char*)weight, N*M*sizeof(float));
  file->write((char*)bias, M*sizeof(float));

}


Connected* Connected::load(char *buf) {

  int para[2] = {0};
  int idx = 0;
  char *token;
  while (buf) {
    token = strtok(NULL, ",");
    para[idx] = atoi(token);
    idx++;
    if(idx > 1)
      break;
  }
  Connected *conn = new Connected(para[0], para[1]);
  return conn;


}
