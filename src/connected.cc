#include "layers.h"

Connected::Connected(int _n, int _m) {
  N = _n;
  M = _m;
}

Connected::~Connected() {

}

void Connected::init() {

#ifdef GPU

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

  random_normal(N*M, weight);
  random_normal(M, bias);

  memset(m_weight, 0, N*M*sizeof(float));
  memset(v_weight, 0, N*M*sizeof(float));
  memset(m_bias, 0, M*sizeof(float));
  memset(v_bias, 0, M*sizeof(float));


#endif

}

void Connected::print() {

  float umem = (float)(4*N*M + 4*M + 2*batch*N)/(1024*1024);

  printf("Conn \t %.2f \t   %d  \t  %d  \n", umem, N, M);

}

void Connected::bias_add() {

  for(int i = 0; i < batch; i++)
    for(int j  = 0; j < M; j++)
      output[i*M+j] += bias[j];
}

void Connected::forward() {  

  gemm_cpu(TRS_N, TRS_N, batch, M, N, 1, input, weight, output);
  bias_add();
}

void Connected::backward(float *delta) {

  gemm_cpu(TRS_N, TRS_T, batch, N, M, 1.0, delta, weight, m_delta);
  gemm_cpu(TRS_T, TRS_N, N, M, batch, 1.0, input, delta, grad_weight);
  row_sum(batch, M, delta, grad_bias);

}

void Connected::update(update_args a) {

#if GPU
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
  file->write(buf, sizeof(buf));
  //cout << buf << endl;
#ifdef GPU
  float *weight_tmp = new float[N*M];
  float *bias_tmp = new float[M];
  gpu_pull_array(weight, weight_tmp, N*M);
  gpu_pull_array(bias, bias_tmp, M);
  file->write((char*)weight_tmp, N*M*sizeof(float));
  file->write((char*)bias_tmp, M*sizeof(float));
  delete []weight_tmp;
  delete []bias_tmp;
#else
  file->write((char*)weight, N*M*sizeof(float));
  file->write((char*)bias, M*sizeof(float));
#endif

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
