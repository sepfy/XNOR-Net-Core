#include "layer/connected.h"

void Connected::Print() {

  float umem = (float)(4*n_*m_ + 4*m_ + 2*batch*n_)/(1024*1024);

  printf("Conn \t %.2f \t 1 x 1 x %d \t\t 1 x 1 x %d  \n", umem, n_, m_);

}

#ifndef GPU
void Connected::Init() {

  weight = new float[n_*m_];
  bias   = new float[m_];
  output = new float[batch*m_];

  if(train_flag_) {
    grad_weight = new float[n_*m_];
    grad_bias = new float[m_];
    delta_ = new float[batch*n_];
    // Adam optimizer 
    m_weight = new float[n_*m_];
    v_weight = new float[n_*m_];
    m_bias = new float[m_];
    v_bias = new float[m_];

    random_normal(n_*m_, weight);
    random_normal(m_, bias);

    memset(m_weight, 0, n_*m_*sizeof(float));
    memset(v_weight, 0, n_*m_*sizeof(float));
    memset(m_bias, 0, m_*sizeof(float));
    memset(v_bias, 0, m_*sizeof(float));
  }

}


void Connected::bias_add() {

  for(int i = 0; i < batch; i++)
    for(int j  = 0; j < m_; j++)
      output[i*m_+j] += bias[j];
}

void Connected::Forward() {  

  gemm_cpu(TRS_n_, TRS_n_, batch, m_, n_, 1, input, weight, output);
  bias_add();
}

void Connected::Backward(float *delta) {

  gemm_cpu(TRS_N, TRS_T, batch, n_, m_, 1.0, delta, weight, delta_);
  gemm_cpu(TRS_T, TRS_N, n_, m_, batch, 1.0, input, delta, grad_weight);
  row_sum(batch, m_, delta, grad_bias);

}

void Connected::Update(UpdateArgs update_args) {

  adam_cpu(n_*m_, weight, grad_weight, m_weight, v_weight, update_args);
  adam_cpu(m_, bias, grad_bias, m_bias, v_bias, update_args);
}

#endif
void Connected::Save(std::fstream *file) {

  char buf[64] = {0};
  sprintf(buf, "Connected,%d,%d", n_, m_);
  file->write(buf, sizeof(buf));
  //cout << buf << endl;
#ifdef GPU
  float *weight_tmp = new float[n_*m_];
  float *bias_tmp = new float[m_];
  gpu_pull_array(weight, weight_tmp, n_*m_);
  gpu_pull_array(bias, bias_tmp, m_);
  file->write((char*)weight_tmp, n_*m_*sizeof(float));
  file->write((char*)bias_tmp, m_*sizeof(float));
  delete []weight_tmp;
  delete []bias_tmp;
#else
  file->write((char*)weight, n_*m_*sizeof(float));
  file->write((char*)bias, m_*sizeof(float));
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
