#include "layer/connected.h"

Connected::Connected(int n, int m) {
  N = n;
  M = m;
}

Connected::~Connected() {

}

void Connected::Print() {

  float umem = (float)(4*N*M + 4*M + 2*batch*N)/(1024*1024);

  printf("Conn \t %.2f \t 1 x 1 x %d \t\t 1 x 1 x %d  \n", umem, N, M);

}

#ifndef GPU
void Connected::Init() {

  weight = new float[N*M];
  bias   = new float[M];
  output = new float[batch*M];

  if(train_flag_) {
    grad_weight = new float[N*M];
    grad_bias = new float[M];
    delta_ = new float[batch*N];
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
  }

}


void Connected::bias_add() {

  for(int i = 0; i < batch; i++)
    for(int j  = 0; j < M; j++)
      output[i*M+j] += bias[j];
}

void Connected::Forward() {  

  gemm_cpu(TRS_N, TRS_N, batch, M, N, 1, input, weight, output);
  bias_add();
}

void Connected::Backward(float *delta) {

  gemm_cpu(TRS_N, TRS_T, batch, N, M, 1.0, delta, weight, delta_);
  gemm_cpu(TRS_T, TRS_N, N, M, batch, 1.0, input, delta, grad_weight);
  row_sum(batch, M, delta, grad_bias);

}

void Connected::Update(UpdateArgs update_args) {

  adam_cpu(N*M, weight, grad_weight, m_weight, v_weight, update_args);
  adam_cpu(M, bias, grad_bias, m_bias, v_bias, update_args);
}

#endif
void Connected::Save(std::fstream *file) {

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
