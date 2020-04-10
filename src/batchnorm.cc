#include "layers.h"

Batchnorm::Batchnorm(int N) {
  this->N = N;
}

Batchnorm::~Batchnorm() {

}

void Batchnorm::init() {


#ifdef GPU
  mean = malloc_gpu(N);

  std = malloc_gpu(N);
  var  = malloc_gpu(N);

  running_mean = malloc_gpu(N);
  running_var  = malloc_gpu(N);
  normal = malloc_gpu(batch*N);
  output = malloc_gpu(batch*N);
  m_delta = malloc_gpu(batch*N);

  xc = malloc_gpu(batch*N);
  dxn = malloc_gpu(batch*N);
  dxc = malloc_gpu(batch*N);
  dvar = malloc_gpu(N);
  dstd = malloc_gpu(N);
  dmu = malloc_gpu(N);

  dgamma = malloc_gpu(N);
  dbeta = malloc_gpu(N);
  gamma = malloc_gpu(N);
  beta = malloc_gpu(N);
  m_gamma = malloc_gpu(N);
  m_beta = malloc_gpu(N);
  v_gamma = malloc_gpu(N);
  v_beta = malloc_gpu(N);

  memset_gpu(gamma, 1.0, N);

#else
  mean = new float[N];
  var  = new float[N];
  std = new float[N];
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


#endif


}

void Batchnorm::print() {
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

void Batchnorm::forward() {

  if(train_flag) {
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

void Batchnorm::backward(float *delta) {


#ifdef GPU
  backward_gpu(delta);
#else

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
#endif
}

void Batchnorm::update(update_args a) {


#ifdef GPU

  if(a.adam) {
    adam_gpu(N, gamma, dgamma, m_gamma, v_gamma, a);
    adam_gpu(N, beta, dbeta, m_beta, v_beta, a);
  }
  else {
    momentum_gpu(N, gamma, dgamma, v_gamma, a);
    momentum_gpu(N, beta, dbeta, v_beta, a);
  }

#else
  adam_cpu(N, gamma, dgamma, m_gamma, v_gamma, a);
  adam_cpu(N, beta, dbeta, m_beta, v_beta, a);
#endif


#if 0

  float m_lr = a.lr * pow(1.0 - pow(a.beta2, a.iter), 0.5) / (1.0 - pow(a.beta1, a.iter));
  for(int i = 0; i < N; i++) {
    m_gamma[i] = (1 - a.beta1)*dgamma[i] + a.beta1*m_gamma[i];
    v_gamma[i] = (1 - a.beta2)*pow(dgamma[i], 2.0) + a.beta2*v_gamma[i];
    m_beta[i] = (1 - a.beta1)*dbeta[i] + a.beta1*m_beta[i];
    v_beta[i] = (1 - a.beta2)*pow(dbeta[i], 2.0) + a.beta2*v_beta[i];
  }

  for(int i = 0; i < N; i++) {
    gamma[i] -= m_lr * m_gamma[i]/(pow(v_gamma[i], 0.5) + a.epsilon);
    beta[i] -= m_lr * m_beta[i]/(pow(v_beta[i], 0.5) + a.epsilon);
  }

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

