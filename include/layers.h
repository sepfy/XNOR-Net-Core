#include <iostream>
#include <vector>
#include "gemm.h"
#include "blas.h"
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include <string.h>

using namespace std;

class Layer {
  public:
    int batch;
    float *input;
    float *output;
    float *m_delta;
    virtual void forward() = 0;
    virtual void backward(float* delta) = 0;
    virtual void update() = 0;
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
   
    Connected(int _n, int _m);
    ~Connected();
    void init(); 
    void forward();
    void backward(float *delta);
    void update();

};

class Sigmoid: public Layer {

  public:
    int N;
    Sigmoid(int _N);
    ~Sigmoid();
    void init();
    void forward();
    void backward(float *delta);
    void update();
};

class SoftmaxWithCrossEntropy : public Layer {

  public:
    int N;
    float *target;

    SoftmaxWithCrossEntropy(int _n, float *_target);
    ~SoftmaxWithCrossEntropy();
    void init();
    void forward();
    void backward(float *delta);
    void update();

};

class Convolution : public Layer {

  public:

    float *col;
    int FW, FH, FC;
    int stride, pad;
    int W, H, C;
    int out_channel;
    int out_w, out_h;
    float *weight, *bias, *out_col, *im;
    float *grad_weight, *grad_bias;

    Convolution(int _W, int _H, int _C,
	int _FW, int _FH, int _FC, int _stride, int _pad);
    ~Convolution();
    void init();
    void forward();
    void backward(float *delta);
    void update();
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
	int _FW, int _FH, int _FC, int _stride, int _pad);
    ~Pooling();
    void init();
    void forward();
    void backward(float *delta);
    void update();
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
    void update();
};
