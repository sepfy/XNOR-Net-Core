#include <iostream>
#include "gemm.h"
#include "blas.h"
class LayerAbs {

  public:
    float *input;
    float *output;
    virtual Tensor forward(Tensor x) = 0;
    virtual Tensor backward(Tensor dx) = 0;
};

class Layer {
  public:
    int batch;
    float *input;
    float *output;
    float *m_delta;
    virtual void forward() = 0;
    virtual void backward(float* delta) = 0;
    virtual void init_var() = 0;
};



class Connected : public Layer {

  public:
    float *weight;
    float *bias;
    float *grad_weight;
    float *grad_bias;

    // W is NxM matrix
    int N;   
    int M;
    int batch;

    Connected(int _batch, int _n, int _m, float *_input) {
      batch = _batch;
      N = _n;
      M = _m;
      input = _input;
      init_var();
    }
    
    ~Connected() {

    }

    void init_var() {

      weight = new float[N*M];
      bias   = new float[M];
      output = new float[batch*M];

      grad_weight = new float[N*M];
      grad_bias = new float[M];
      m_delta = new float[batch*N];

      srand(time(NULL));
      for(int j = 0; j < M; j++) {
        bias[j] = 0.1*((float) rand()/(RAND_MAX + 1.0) - 0.5);
        for(int i = 0; i < N; i++) {
          weight[i*M+j] = 0.1*((float) rand()/(RAND_MAX + 1.0) -0.5);
        }
      }
    }
  
    void forward() {  
      
      memset(output, 0, batch*M*sizeof(float));
      bias_add(batch, M, output, bias);
      gemm(batch, M, N, 1, input, weight, output);
    }

    void backward(float *delta) {

      memset(grad_weight, 0, N*M*sizeof(float));
      memset(m_delta, 0, batch*N*sizeof(float));

      gemm_ta(N, M, batch, 1.0, input, delta, grad_weight);
      gemm_tb(batch, N, M, 1.0, delta, weight, m_delta);
      row_sum(batch, M, delta, grad_bias);
    }

    void update() {
      mat_scalar(N, M, grad_weight, 1.0, grad_weight);
      mat_minus(N, M, weight, grad_weight, weight);

      mat_scalar(1, M, grad_bias, 1.0, grad_bias);
      mat_minus(1, M, bias, grad_bias, bias);
    }

};

class SigmoidA: public Layer {

  public:
    int N;
    SigmoidA(int _batch, int _N, float *_input) {
      batch = _batch;
      N = _N;
      input = _input;
      init_var();  
    }
    ~SigmoidA() {}
    void init_var() {
      output = new float[batch*N];
      m_delta = new float[batch*N];
    }
    void forward() {
      for(int i = 0; i < batch; i++) 
        for(int j = 0; j < N; j++) 
          output[i*N+j] = 1.0/(1.0 + exp(-1.0*(input[i*N+j])));
    }

    void backward(float *delta) {
      for(int i = 0; i < batch; i++) 
        for(int j = 0; j < N; j++) 
          m_delta[i*N+j] = delta[i*N+j]*(1.0 - output[i*N+j])*output[i*N+j];          }
};

class SoftmaxWithCrossEntropy : public Layer {

  public:
    int N;
    float *target;

    SoftmaxWithCrossEntropy(int _batch, int _n, float *_target, float *_input) {
      batch = _batch;
      N = _n;
      target = _target;
      input = _input;
      init_var();
    }
    void init_var() {
      output = new float[batch*N];
      m_delta = new float[batch*N];
    }

    ~SoftmaxWithCrossEntropy() {

    }    

    void forward() {

      for(int i = 0; i < batch; i++) {
        float tmp = 0;
        float max = 0;
        for(int j = 0; j < N; j++) 
          if(input[i*N+j] > max)
            max = input[i*N+j];
        
        for(int j = 0; j < N; j++) {
          output[i*N+j] = exp(input[i*N+j] - max);
          tmp += output[i*N+j];
        }
        for(int j = 0; j < N; j++) 
          output[i*N+j] /= tmp;
      }
    }
    void backward(float *delta) {
      mat_minus(batch, N, output, target, m_delta);  
      mat_scalar(batch, N, m_delta, 1.0/(float)batch, m_delta);
    }
};

class Relu : public Layer {
  public:
    Relu() {

    }
    
    ~Relu() {

    }
};

class Affine : public LayerAbs {

  public:
    Tensor W;
    Tensor b;
    Tensor dW;
    Tensor db;
    Tensor x;

    Affine(Tensor _W, Tensor _b): W(_W), b(_b), dW(_W), db(_b) {}

    Tensor forward(Tensor _x) {
      x = _x;
      return x*W + b;
    }

    Tensor backward(Tensor dout) {

      for(int i = 0; i < db.shape[1]; i++) {
        db.value[0][i] = 0;
        for(int j = 0; j < dout.shape[0]; j++)
          db.value[0][i] += dout.value[j][i];
      }
      dW = x.T()*dout;
      
      dout = dout*W.T();
      return dout;
    }


};




//6000x10 *10*10 -> 6000*10

class Sigmoid: public LayerAbs {

  public:
    Tensor Y;
 
    Tensor forward(Tensor t) {
      Y.init(t.shape[0], t.shape[1]);
      for(int i = 0; i < t.shape[0]; i++) {
        for(int j = 0; j < t.shape[1]; j++) {
          Y.value[i][j] = 1.0/(1.0 + exp(-1.0*(t.value[i][j])));
        }
      }
      return Y;
    } 
//6000x10
// Y: 6000x10
// dout: 
    Tensor backward(Tensor dout) {
      Tensor _dout(dout.shape[0], dout.shape[1], 0.0);  
      
      for(int i = 0; i < dout.shape[0]; i++) 
        for(int j = 0; j < dout.shape[1]; j++) 
          _dout.value[i][j] = dout.value[i][j]*(1.0 - Y.value[i][j])*Y.value[i][j];            
      return _dout;
    }
};

class Softmax: public LayerAbs {
  
  public:
    Tensor Y;
    Softmax(Tensor _Y): Y(_Y) {}
    Tensor forward(Tensor t) {
      Tensor t1(t.shape[0], t.shape[1]);
      for(int i = 0; i < t.shape[0]; i++) {
        float tmp = 0;
        float max = 0;
        for(int j = 0; j < t.shape[1]; j++) 
          if(t.value[i][j] > max)
            max = t.value[i][j];
        
        for(int j = 0; j < t.shape[1]; j++) {
          t1.value[i][j] = exp(t.value[i][j] - max);
          tmp += t1.value[i][j];
        }
        for(int j = 0; j < t.shape[1]; j++) { 
          t1.value[i][j] /= tmp;
        }
      }
      return t1;
    }

//6000x10
    Tensor backward(Tensor dout) {
      float batch_size = (float)this->Y.shape[0];
      return (dout - this->Y)*(1.0/batch_size);
    }
  
};




