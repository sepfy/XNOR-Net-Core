#include <iostream>
#include <vector>
#include "gemm.h"
#include "blas.h"
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include <string.h>

//#define XNOR_NET
using namespace std;

class Layer {
  public:
    int batch;
    float *input;
    float *output;
    float *m_delta;
    virtual void forward() = 0;
    virtual void backward(float* delta) = 0;
    virtual void update(float lr) = 0;
    virtual void init() = 0;
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
  
    // Adam optimizer
    float beta1 = 0.9;
    float beta2 = 0.999;
    float *m_weight;
    float *v_weight;
    float *m_bias;
    float *v_bias;
    float iter = 0.0;
    float eplson = 1.0e-7;
 
    Connected(int _n, int _m);
    ~Connected();
    void init(); 
    void forward();
    void backward(float *delta);
    void update(float lr);

};

class Sigmoid: public Layer {

  public:
    int N;
    Sigmoid(int _N);
    ~Sigmoid();
    void init();
    void forward();
    void backward(float *delta);
    void update(float lr);
};

class SoftmaxWithCrossEntropy : public Layer {

  public:
    int N;

    SoftmaxWithCrossEntropy(int _n);
    ~SoftmaxWithCrossEntropy();
    void init();
    void forward();
    void backward(float *delta);
    void update(float lr);

};

class Convolution : public Layer {

  public:

    float *col;
    int FW, FH, FC;
    int stride, pad;

    int W, H, C;
    int out_channel;
    int out_w, out_h;
    int col_size;
    int im_size;
    int weight_size;
    int input_size;

    float *weight, *bias, *out_col, *im;
    float *grad_weight, *grad_bias;

    // Adam optimizer
    float beta1 = 0.9;
    float beta2 = 0.999;
    float *m_weight;
    float *v_weight;
    float *m_bias;
    float *v_bias;
    float iter = 0.0;
    float eplson = 1.0e-7;

    Convolution(int _W, int _H, int _C,
	int _FW, int _FH, int _FC, int _stride, bool _pad);
    ~Convolution();
    void init();
    void forward();
    void backward(float *delta);
    void update(float lr);

#ifdef XNOR_NET
    float *binary_weight;
    float *binary_input;
    float *avg_filter;
    float *avg_col;
    float *k_filter;
    float *k_output;
    void swap_weight();
    float binarize_weight();
    void binarize_input();
#endif
};

class Pooling : public Layer {

  public:

    float *col;
    int FW, FH, FC;
    int stride, pad;
    int W, H, C;
    int out_channel;
    int out_w, out_h;
    float *weight, *bias, *out_col, *im;
    float *grad_weight;
    float *delta_col;
    vector<int> max_seq;
    Pooling(int _W, int _H, int _C,
	int _FW, int _FH, int _FC, int _stride, bool _pad);
    ~Pooling();
    void init();
    void forward();
    void backward(float *delta);
    void update(float lr);
};

class Relu : public Layer {
  public:
    int N;
    vector<int> mask;
    Relu(int _N);
    ~Relu();
    void init();
    void forward();
    void backward(float *delta);
    void update(float lr);
};
