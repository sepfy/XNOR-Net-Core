#include "layer/batchnorm.h"

Batchnorm::Batchnorm(int N) {
  this->N = N;
}

Batchnorm::~Batchnorm() {

#ifdef GPU

#else
  delete []std;
  delete []running_mean;
  delete []running_var;
  delete []normal;
  delete []gamma;
  delete []beta;
  delete []output;

  if(train_flag_) {
    delete []mean;
    delete []var;
    delete []delta_;
    delete []dxn; 
    delete []dxc; 
    delete []dvar; 
    delete []dstd; 
    delete []dmu; 
    delete []dgamma; 
    delete []dbeta; 
    delete []m_gamma; 
    delete []m_beta; 
    delete []v_gamma; 
    delete []v_beta; 
  }

#endif
}
void Batchnorm::Print() {
  float umem = (float)(15*N + 6*batch*N)/(1024*1024);

  printf("BN \t %.2f\n", umem);

}


void Batchnorm::get_mean() {

  memset(mean, 0, N*sizeof(float));
  for(int i = 0; i < batch; i++)
    for(int j = 0; j < N; j++)
      mean[j] += input[i*N+j]/(float)batch;
}

void Batchnorm::get_variance() {

  memset(var, 0, N*sizeof(float));
  for(int i = 0; i < batch; i++)
    for(int j = 0; j < N; j++)
      var[j] += (pow(input[i*N+j] - mean[j], 2.0))/(float)batch;


}

void Batchnorm::normalize() {
  for(int i = 0; i < batch; i++)
    for(int j = 0; j < N; j++)
      normal[i*N+j] = (input[i*N+j] - mean[j])/pow(var[j] + epsilon, 0.5);

  for(int j = 0; j < N; j++) {
    running_mean[j] = momentum*running_mean[j] + (1.0 - momentum)*mean[j];
    running_var[j] = momentum*running_var[j] + (1.0 - momentum)*var[j];
  }

}

void Batchnorm::scale_and_shift() {

  for(int i = 0; i < batch; i++)
    for(int j = 0; j < N; j++)
      output[i*N+j] = gamma[j]*normal[i*N+j] + beta[j];
}

#ifndef GPU

void Batchnorm::Init() {

  std = new float[N];
  running_mean = new float[N];
  running_var  = new float[N];
  gamma = new float[N];
  beta = new float[N];
  output = new float[batch*N];

  if(!runtime)
    normal = new float[batch*N];

  if(train_flag_) {
    mean = new float[N];
    var  = new float[N];
    delta_ = new float[batch*N];
    dxn = new float[batch*N];
    dxc = new float[batch*N];
    dvar = new float[N];
    dstd = new float[N];
    dmu = new float[N];

    dgamma = new float[N];
    dbeta = new float[N];
    m_gamma = new float[N];
    m_beta = new float[N];
    v_gamma = new float[N];
    v_beta = new float[N];

    for(int i = 0; i < N; i++) {
      gamma[i] = 1.0;
      beta[i] = 0.0;

      m_beta[i] = 0.0;
      v_beta[i] = 0.0;
      m_gamma[i] = 0.0;
      m_gamma[i] = 0.0;

      running_mean[i] = 0.0;
      running_var[i] = 0.0;
    }

  }


}



void Batchnorm::Forward() {

  if(train_flag_) {
    get_mean();
    get_variance();
    normalize();
    scale_and_shift();
  }
  else {


    if(runtime) {
      for(int i = 0; i < batch; i++)
        for(int j = 0; j < N; j++)
          output[i*N+j] = gamma[j]*(input[i*N+j] - running_mean[j])/std[j] + beta[j];
    }
    else {
      for(int i = 0; i < batch; i++)
        for(int j = 0; j < N; j++)
          normal[i*N+j] = (input[i*N+j] - running_mean[j])/pow(running_var[j] + epsilon, 0.5);
      scale_and_shift();
    }
  }


}

void Batchnorm::Backward(float *delta) {

  memset(dbeta, 0 , N*sizeof(float));
  memset(dgamma, 0 , N*sizeof(float));

  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      dbeta[j] += delta[i*N+j];
      dgamma[j] += normal[i*N+j]*delta[i*N+j];
    }
  }

  // Step8
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      dxn[i*N+j] = gamma[j]*delta[i*N+j];
    }
  }
  

  // Step2+7
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      dxc[i*N+j] = dxn[i*N+j]/pow(var[j] + epsilon, 0.5);
    }
  }


  
  // Step6+7
  memset(dstd, 0, sizeof(float)*N);
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      dstd[j] -= dxn[i*N+j]*(input[i*N+j]-mean[j])/(var[j] + epsilon);
    }
  }

  // Step5
  for(int j = 0; j < N; j++) 
    dvar[j] = 0.5*dstd[j]/pow(var[j] + epsilon, 0.5);

  // Step3+4
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      dxc[i*N+j] += (2.0/(float)batch)*(input[i*N+j] - mean[j])*dvar[j];
    }
  }


  // Step1
  memset(dmu, 0, N*sizeof(float));
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      dmu[j] += dxc[i*N+j];
    }
  }

  // Step 0
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      delta_[i*N+j] = dxc[i*N+j] - dmu[j]/(float)batch;

    }
  }
}


void Batchnorm::Update(UpdateArgs a) {

  adam_cpu(N, gamma, dgamma, m_gamma, v_gamma, a);
  adam_cpu(N, beta, dbeta, m_beta, v_beta, a);

}
#endif

void Batchnorm::Save(std::fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Batchnorm,%d", N);
  file->write(buf, sizeof(buf));

#ifdef GPU
  float *mean_tmp = new float[N];
  float *var_tmp = new float[N];
  float *gamma_tmp = new float[N];
  float *beta_tmp = new float[N];
  gpu_pull_array(running_mean, mean_tmp, N);
  gpu_pull_array(running_var, var_tmp, N);
  gpu_pull_array(gamma, gamma_tmp, N);
  gpu_pull_array(beta, beta_tmp, N);
  file->write((char*)mean_tmp, N*sizeof(float));
  file->write((char*)var_tmp, N*sizeof(float));
  file->write((char*)gamma_tmp, N*sizeof(float));
  file->write((char*)beta_tmp, N*sizeof(float));
  delete []mean_tmp;
  delete []var_tmp;
  delete []gamma_tmp;
  delete []beta_tmp;
#else
  file->write((char*)running_mean, N*sizeof(float));
  file->write((char*)running_var, N*sizeof(float));
  file->write((char*)gamma, N*sizeof(float));
  file->write((char*)beta, N*sizeof(float));
#endif
}

Batchnorm* Batchnorm::load(char *buf) {
  int para = 0;
  char *token;
  token = strtok(NULL, ",");
  para = atoi(token);
  Batchnorm *bn = new Batchnorm(para);
  return bn;
}

