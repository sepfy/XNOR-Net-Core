#include <iostream>
#include "gemm.h"
#include "blas.h"
#include "utils.h"
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

class Sigmoid: public Layer {

  public:
    int N;
    Sigmoid(int _batch, int _N, float *_input) {
      batch = _batch;
      N = _N;
      input = _input;
      init_var();  
    }
    ~Sigmoid() {}
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

class Convolution : public Layer {

  public:

    float *col;
    int batch;
    int FW, FH, FC;
    int stride, pad;
    int W, H, C;
    int channel_out;
    int out_w, out_h;
    float *weight, *bias, *out_col, *im;
    Convolution(int _batch, int _W, int _H, int _C, int _FW, int _FH, int _FC,
                int _stride, int _pad, float* _input) {
      batch = _batch;
      W = _W;
      H = _H;
      C = _C;
      FW = _FW;
      FH = _FH;
      FC = _FC;
      stride = _stride;
      pad = _pad;
      out_w = (W + 2*pad - FW)/stride + 1;
      out_h = (H + 2*pad - FH)/stride + 1;
      channel_out = FW*FH*C;
      input = _input;
      init_var();
    }
    ~Convolution() {

    }
    void init_var() {
      col = new float[out_w*out_h*channel_out];
      output = new float[batch*out_w*out_h*FC];
      out_col = new float[out_w*out_h*FC];
      weight = new float[channel_out*FC];
      im = new float[H*W*C];
     // im_out = new float[out_w*out_h*FC];
    }
    void forward() {
      int size = out_w*out_h*FC;
      int im_size = H*W*C;
      for(int i = 0; i < batch-3; i++) {
        memcpy(im, input + i*im_size, im_size*sizeof(float));
        im2col(W, H, C, FW, FH, FC, stride, pad, im, col);
        gemm(out_w*out_h, FC, channel_out, 1, col, weight, out_col);
        //col2im(W, H, C, FW, FH, FC, stride, pad, im, col);
        memcpy(output + i*size, out_col, size*sizeof(float));
      }
    }
    void backward(float *delta) {

    }


};

class Relu : public Layer {
  public:
    Relu() {

    }
    
    ~Relu() {

    }

    void forward() {

    }

    void backward(float *delta) {

    }
};

