#include "layers.h"

Batchnorm::Batchnorm(int _N) {
  N = _N;
}

Batchnorm::~Batchnorm() {

}

void Batchnorm::init() {

  mean = new float[N];
  var  = new float[N];
  running_mean = new float[N];
  running_var  = new float[N];
  normal = new float[batch*N];
  output = new float[batch*N];
  m_delta = new float[batch*N];

  dxn = new float[batch*N];
  dxc = new float[batch*N];
  dvar = new float[N];
  dstd = new float[N];
  dmu = new float[N];

  dgamma = new float[N];
  dbeta = new float[N];
  gamma = new float[N];
  beta = new float[N];
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

void Batchnorm::forward() {


  if(train_flag) {
    memset(mean, 0, N*sizeof(float));
    memset(var, 0, N*sizeof(float));

    for(int i = 0; i < batch; i++)
      for(int j = 0; j < N; j++)
        mean[j] += input[i*N+j];
  
    for(int j = 0; j < N; j++)
      mean[j] /= (float)batch;

    for(int i = 0; i < batch; i++)
      for(int j = 0; j < N; j++)
        var[j] += pow(input[i*N+j] - mean[j], 2.0);

    for(int j = 0; j < N; j++)
      var[j] /= (float)batch;


    for(int i = 0; i < batch; i++)
      for(int j = 0; j < N; j++) {
        normal[i*N+j] = (input[i*N+j] - mean[j])/pow(var[j] + epsilon, 0.5);
        output[i*N+j] = gamma[j]*normal[i*N+j] + beta[j];
      }

    for(int j = 0; j < N; j++) {
      running_mean[j] = momentum*running_mean[j] + (1-momentum)*mean[j];
      running_var[j] = momentum*running_var[j] + (1-momentum)*var[j];
    }
  }
  else {

    for(int i = 0; i < batch; i++)
      for(int j = 0; j < N; j++) {
        normal[i*N+j] = (input[i*N+j] - running_mean[j])/pow(running_var[j] + epsilon, 0.5);
        output[i*N+j] = gamma[j]*normal[i*N+j] + beta[j];
      }

  }





}

void Batchnorm::backward(float *delta) {

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
      m_delta[i*N+j] = dxc[i*N+j] - dmu[j]/(float)batch;

    }
  }
}

void Batchnorm::update(float lr) {

  iter++;
  float m_lr = lr * pow(1.0 - pow(beta2, iter), 0.5) / (1.0 - pow(beta1, iter));
  for(int i = 0; i < N; i++) {
    m_gamma[i] = (1 - beta1)*dgamma[i] + beta1*m_gamma[i];
    v_gamma[i] = (1 - beta2)*pow(dgamma[i], 2.0) + beta2*v_gamma[i];
    m_beta[i] = (1 - beta1)*dbeta[i] + beta1*m_beta[i];
    v_beta[i] = (1 - beta2)*pow(dbeta[i], 2.0) + beta2*v_beta[i];
  }

  for(int i = 0; i < N; i++) {
    gamma[i] -= m_lr * m_gamma[i]/(pow(v_gamma[i], 0.5) + epsilon);
    beta[i] -= m_lr * m_beta[i]/(pow(v_beta[i], 0.5) + epsilon);
  }

#if 0
  for(int i = 0; i < N; i++) {
    gamma[i] -= lr*dgamma[i];
    beta[i] -= lr*dbeta[i];
  } 
#endif

}

void Batchnorm::save(fstream *file) {
  char buf[64] = {0};
  sprintf(buf, "Batchnorm,%d", N);
  file->write(buf, sizeof(buf));
}

Batchnorm* Batchnorm::load(char *buf) {
  int para = 0;
  char *token;
  token = strtok(NULL, ",");
  para = atoi(token);
  Batchnorm *bn = new Batchnorm(para);
  return bn;
}

