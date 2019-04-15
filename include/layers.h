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
    int batch;

    Connected(int _batch, int _n, int _m, float *_input);
    ~Connected();
    void init(); 
    void forward();
    void backward(float *delta);
    void update();

};

class Sigmoid: public Layer {

  public:
    int N;
    Sigmoid(int _batch, int _N, float *_input);
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

    SoftmaxWithCrossEntropy(int _batch, int _n, float *_target, float *_input);
    ~SoftmaxWithCrossEntropy();
    void init();
    void forward();
    void backward(float *delta);
    void update();

};

class Convolution : public Layer {

  public:

    float *col;
    int batch;
    int FW, FH, FC;
    int stride, pad;
    int W, H, C;
    int out_channel;
    int out_w, out_h;
    float *weight, *bias, *out_col, *im;
    float *grad_weight;

    Convolution(int _batch, int _W, int _H, int _C,
	int _FW, int _FH, int _FC, int _stride, int _pad, float* _input);
    ~Convolution();
    void init();
    void forward();
    void backward(float *delta);
    void update();
};

class Pooling : public Layer {

  public:

    float *col;
    int batch;
    int FW, FH, FC;
    int stride, pad;
    int W, H, C;
    int out_channel;
    int out_w, out_h;
    float *weight, *bias, *out_col, *im;
    float *grad_weight;

    Pooling(int _batch, int _W, int _H, int _C,
	int _FW, int _FH, int _FC, int _stride, int _pad, float* _input);
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
    Relu(int _batch, int _N, float *_input);
    ~Relu();
    void init();
    void forward();
    void backward(float *delta);
    void update();
};
