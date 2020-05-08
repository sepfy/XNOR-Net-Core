#include "layer/batchnorm.h"

void Batchnorm::Print() {
  float umem = (float)(15*n_ + 6*batch*n_)/(1024*1024);

  printf("Bn_ \t %.2f\n", umem);

}

#ifndef GPU
void Batchnorm::GetMean() {

  memset(mean, 0, n_*sizeof(float));
  for(int i = 0; i < batch; i++)
    for(int j = 0; j < n_; j++)
      mean[j] += input[i*n_+j]/(float)batch;
}

void Batchnorm::GetVariance() {

  memset(var, 0, n_*sizeof(float));
  for(int i = 0; i < batch; i++)
    for(int j = 0; j < n_; j++)
      var[j] += (pow(input[i*n_+j] - mean[j], 2.0))/(float)batch;


}

void Batchnorm::Normalize() {
  for(int i = 0; i < batch; i++)
    for(int j = 0; j < n_; j++)
      normal[i*n_+j] = (input[i*n_+j] - mean[j])/pow(var[j] + epsilon, 0.5);

  for(int j = 0; j < n_; j++) {
    running_mean[j] = momentum*running_mean[j] + (1.0 - momentum)*mean[j];
    running_var[j] = momentum*running_var[j] + (1.0 - momentum)*var[j];
  }

}

void Batchnorm::ScaleAndShift() {

  for(int i = 0; i < batch; i++)
    for(int j = 0; j < n_; j++)
      output[i*n_+j] = gamma[j]*normal[i*n_+j] + beta[j];
}


void Batchnorm::Init() {

  std = new float[n_];
  running_mean = new float[n_];
  running_var  = new float[n_];
  gamma = new float[n_];
  beta = new float[n_];
  output = new float[batch*n_];

  if(!runtime)
    normal = new float[batch*n_];

  if(train_flag_) {
    mean = new float[n_];
    var  = new float[n_];
    delta_ = new float[batch*n_];
    dxn = new float[batch*n_];
    dxc = new float[batch*n_];
    dvar = new float[n_];
    dstd = new float[n_];
    dmu = new float[n_];

    dgamma = new float[n_];
    dbeta = new float[n_];
    m_gamma = new float[n_];
    m_beta = new float[n_];
    v_gamma = new float[n_];
    v_beta = new float[n_];

    for(int i = 0; i < n_; i++) {
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
    GetMean();
    GetVariance();
    Normalize();
    ScaleAndShift();
  }
  else {


    if(runtime) {
      for(int i = 0; i < batch; i++)
        for(int j = 0; j < n_; j++)
          output[i*n_+j] = gamma[j]*(input[i*n_+j] - running_mean[j])/std[j] + beta[j];
    }
    else {
      for(int i = 0; i < batch; i++)
        for(int j = 0; j < n_; j++)
          normal[i*n_+j] = (input[i*n_+j] - running_mean[j])/pow(running_var[j] + epsilon, 0.5);
      ScaleAndShift();
    }
  }


}

void Batchnorm::Backward(float *delta) {

  memset(dbeta, 0 , n_*sizeof(float));
  memset(dgamma, 0 , n_*sizeof(float));

  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < n_; j++) {
      dbeta[j] += delta[i*n_+j];
      dgamma[j] += normal[i*n_+j]*delta[i*n_+j];
    }
  }

  // Step8
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < n_; j++) {
      dxn[i*n_+j] = gamma[j]*delta[i*n_+j];
    }
  }
  

  // Step2+7
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < n_; j++) {
      dxc[i*n_+j] = dxn[i*n_+j]/pow(var[j] + epsilon, 0.5);
    }
  }


  
  // Step6+7
  memset(dstd, 0, sizeof(float)*n_);
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < n_; j++) {
      dstd[j] -= dxn[i*n_+j]*(input[i*n_+j]-mean[j])/(var[j] + epsilon);
    }
  }

  // Step5
  for(int j = 0; j < n_; j++) 
    dvar[j] = 0.5*dstd[j]/pow(var[j] + epsilon, 0.5);

  // Step3+4
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < n_; j++) {
      dxc[i*n_+j] += (2.0/(float)batch)*(input[i*n_+j] - mean[j])*dvar[j];
    }
  }


  // Step1
  memset(dmu, 0, n_*sizeof(float));
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < n_; j++) {
      dmu[j] += dxc[i*n_+j];
    }
  }

  // Step 0
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < n_; j++) {
      delta_[i*n_+j] = dxc[i*n_+j] - dmu[j]/(float)batch;

    }
  }
}


void Batchnorm::Update(UpdateArgs a) {

  adam_cpu(n_, gamma, dgamma, m_gamma, v_gamma, a);
  adam_cpu(n_, beta, dbeta, m_beta, v_beta, a);

}
#endif

void Batchnorm::Save(std::fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Batchnorm,%d", n_);
  file->write(buf, sizeof(buf));

#ifdef GPU
  float *mean_tmp = new float[n_];
  float *var_tmp = new float[n_];
  float *gamma_tmp = new float[n_];
  float *beta_tmp = new float[n_];
  gpu_pull_array(running_mean, mean_tmp, n_);
  gpu_pull_array(running_var, var_tmp, n_);
  gpu_pull_array(gamma, gamma_tmp, n_);
  gpu_pull_array(beta, beta_tmp, n_);
  file->write((char*)mean_tmp, n_*sizeof(float));
  file->write((char*)var_tmp, n_*sizeof(float));
  file->write((char*)gamma_tmp, n_*sizeof(float));
  file->write((char*)beta_tmp, n_*sizeof(float));
  delete []mean_tmp;
  delete []var_tmp;
  delete []gamma_tmp;
  delete []beta_tmp;
#else
  file->write((char*)running_mean, n_*sizeof(float));
  file->write((char*)running_var, n_*sizeof(float));
  file->write((char*)gamma, n_*sizeof(float));
  file->write((char*)beta, n_*sizeof(float));
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

