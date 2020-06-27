#include "layer/batchnorm.h"

void Batchnorm::Print() {
  printf("Batchnorm\n");
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

  std = new float[channel_];
  running_mean = new float[channel_];
  running_var  = new float[channel_];
  gamma = new float[channel_];
  beta = new float[channel_];
  output = new float[batch*spatial_*channel_];

  if(!runtime)
    normal = new float[batch*channel_*spatial_];

  if(train_flag_) {
    mean = new float[channel_];
    var  = new float[channel_];
    delta_ = new float[batch*channel_*spatial_];
    dxn = new float[batch*channel_*spatial_];
    dxc = new float[batch*channel_*spatial_];
    dvar = new float[channel_];
    dstd = new float[channel_];
    dmu = new float[channel_];

    dgamma = new float[channel_];
    dbeta = new float[channel_];
    m_gamma = new float[channel_];
    m_beta = new float[channel_];
    v_gamma = new float[channel_];
    v_beta = new float[channel_];

    for(int i = 0; i < channel_; i++) {
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


void Batchnorm::LoadParams(std::fstream *rfile, int batch) {

  this->batch = batch;
  train_flag_ = false;
  runtime = true;
  Init();
  rfile->read((char*)running_mean, channel_*sizeof(float));
  rfile->read((char*)running_var, channel_*sizeof(float));
  rfile->read((char*)gamma, channel_*sizeof(float));
  rfile->read((char*)beta, channel_*sizeof(float));
  for(int i = 0; i < channel_; i++)
      std[i] = pow(running_var[i] + epsilon, 0.5);

}


void Batchnorm::Save(std::fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Batchnorm,%d,%d", spatial_, channel_);
  file->write(buf, sizeof(buf));

  file->write((char*)running_mean, channel_*sizeof(float));
  file->write((char*)running_var, channel_*sizeof(float));
  file->write((char*)gamma, channel_*sizeof(float));
  file->write((char*)beta, channel_*sizeof(float));
}

#endif


Batchnorm* Batchnorm::load(char *buf) {

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
  Batchnorm *bn = new Batchnorm(para[0], para[1]);
  return bn;
}

