#include "layers.h"

Batchnorm::Batchnorm(int _N) {
  N = _N;
}

Batchnorm::~Batchnorm() {

}

void Batchnorm::init() {

  mean = new float[N];
  var  = new float[N];
  normal = new float[batch*N];
  output = new float[batch*N];
  m_delta = new float[batch*N];

  dxn = new float[batch*N];
  dxc = new float[batch*N];
  dvar = new float[N];
  dstd = new float[N];
  dmu = new float[N];
  
}

void Batchnorm::forward() {

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
      normal[i*N+j] = (input[i*N+j] - mean[j])/pow(var[j] + epslon, 0.5);
      output[i*N+j] = gamma*normal[i*N+j] + beta;
    }

}

void Batchnorm::backward(float *delta) {

  // Step8
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      dxn[i*N+j] = gamma*delta[i*N+j];
    }
  }
  

  // Step2+7
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      dxc[i*N+j] = dxn[i*N+j]/pow(var[j] + epslon, 0.5);
    }
  }


  
  // Step6+7
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      dstd[j] -= dxn[i*N+j]*(input[i*N+j]-mean[j])/(var[j] + epslon);
    }
  }

  // Step5
  for(int j = 0; j < N; j++) 
    dvar[j] = 0.5*dstd[j]/pow(var[j] + epslon, 0.5);

  // Step3+4
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      dxc[i*N+j] += (2.0/(float)batch)*(input[i*N+j] - mean[j])*dvar[j];
    }
  }


  // Step1
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
}

void Batchnorm::save(FILE *fp) {

}

