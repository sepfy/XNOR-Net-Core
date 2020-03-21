#include <iostream>
#include <vector>
#include "gemm.h"
#include "blas.h"
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include <string.h>
#include <fstream>
#include "binary.h"
#include "optimizer.h"
#ifdef GPU
#include "gpu.h"
#endif

#define XNOR_NET
using namespace std;

class Layer {
  public:
    int batch;
    bool train_flag = true;
    float *input;
    float *output;
    float *m_delta;
    size_t shared_size = 0;
    float *shared;
    virtual void forward() = 0;
    virtual void backward(float* delta) = 0;

#ifdef GPU
    virtual void forward_gpu() = 0;
    virtual void backward_gpu(float* delta) = 0;
#endif

    virtual void update(update_args a) = 0;
    virtual void init() = 0;
    virtual void print() = 0;
    virtual void save(fstream *file) = 0;
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
    float epsilon = 1.0e-7;
 
    Connected(int _n, int _m);
    ~Connected();
    void init(); 
    void print();
    void forward();
#if GPU
    void forward_gpu();
    void backward_gpu(float* delta);
#endif
    void bias_add();
    void backward(float *delta);
    void update(update_args a);
    void save(fstream *file);
    static Connected* load(char *buf);

};
/*
class Sigmoid: public Layer {

  public:
    int N;
    Sigmoid(int _N);
    ~Sigmoid();
    void init();
    void forward();
    void backward(float *delta);
    void update(update_args a);
    void save(fstream *file);
};
*/
class SoftmaxWithCrossEntropy : public Layer {

  public:
    int N;

    SoftmaxWithCrossEntropy(int _n);
    ~SoftmaxWithCrossEntropy();
    void init();
    void print();
    void forward();
    void backward(float *delta);
    void update(update_args a);
#if GPU
    void forward_gpu();
    void backward_gpu(float* delta);
#endif
    void save(fstream *file);
    static SoftmaxWithCrossEntropy* load(char *buf);

};

class Convolution : public Layer {

  public:
    bool xnor = true;
    float *col;
    int FW, FH, FC;
    int stride, pad;
    float *delta_col;
    int W, H, C;
    int out_channel;
    int out_w, out_h;
    int col_size;
    int im_size;
    int weight_size;
    int bias_size;
    int input_size;

    bool runtime = false;

    float *weight, *bias, *out_col, *im;
    float *grad_weight, *grad_bias;
    float *mean;
    // Adam optimizer
    float beta1 = 0.9;
    float beta2 = 0.999;
    float *m_weight;
    float *v_weight;
    float *m_bias;
    float *v_bias;
    float iter = 0.0;
    float epsilon = 1.0e-7;

    Convolution(int _W, int _H, int _C,
	int _FW, int _FH, int _FC, int _stride, bool _pad);
    ~Convolution();
    void init();
    void print();
    
    void bias_add();
    void forward_xnor();
    void forward_full();
    float* backward_xnor(float *delta);
    float* backward_full(float *delta);
    void forward();
    void backward(float *delta);
    void update(update_args a);
    void save(fstream *file);
    static Convolution* load(char *buf);

#ifdef GPU
    void forward_gpu();
    void backward_gpu(float* delta);
    void bias_add_gpu();
    void binarize_input_gpu();
    void binarize_weight_gpu();
#endif

#ifdef XNOR_NET
    float *binary_weight;
    float *binary_input;
    float *avg_filter;
    float *avg_col;
    float *k_filter;
    float *k_output;
    Bitset *bitset_outcol, *bitset_weight;
    void swap_weight();
    float binarize_weight();
    void binarize_input();
#endif
};


class AvgPool : public Layer {

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
    float *indexes;
    AvgPool(int _W, int _H, int _C,
	int _FW, int _FH, int _FC, int _stride, bool _pad);
    ~AvgPool();
    void init();
    void print();
    void forward();
    void backward(float *delta);
    void update(update_args a);
    void save(fstream *file);
    static AvgPool* load(char *buf);

#ifdef GPU
    void forward_gpu();
    void backward_gpu(float *delta);
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
    float *indexes;
    Pooling(int _W, int _H, int _C,
	int _FW, int _FH, int _FC, int _stride, bool _pad);
    ~Pooling();
    void init();
    void print();
    void forward();
    void backward(float *delta);
    void update_gpu(update_args a);
    void update(update_args a);
    void save(fstream *file);
    static Pooling* load(char *buf);

#ifdef GPU
    void forward_gpu();
    void backward_gpu(float *delta);
#endif

};

enum ACT{
  RELU,
  LEAKY,
  SIGMD,
  NUM_TYPE
};

class Activation : public Layer {
  public:
    
    ACT activation;
    void relu_activate();
    void leaky_activate();
    void relu_backward(float *delta);
    void leaky_backward(float *delta);

    int N;
    float *cut;
    vector<int> mask;
    Activation(int _N, ACT act);
    ~Activation();
    void init();
    void print();
    void forward();
    void backward(float *delta);
    void update(update_args a);
    void save(fstream *file);
    static Activation* load(char *buf);

#ifdef GPU
    void backward_gpu(float *delta);
    void forward_gpu();
    void relu_activate_gpu();
    void leaky_activate_gpu();
    void relu_backward_gpu(float *delta);
    void leaky_backward_gpu(float *delta);
#endif
};



class Batchnorm : public Layer {
  public:
    int N;
    float iter = 0.0;
    float *mean, *var, *std, *running_mean, *running_var;
    float *normal;
    float epsilon = 1.0e-7;
    float *gamma, *beta, *dgamma, *dbeta;
    float *m_gamma, *m_beta, *v_gamma, *v_beta;
    float *dxn;
    float *dxc;
    float *dvar;
    float *dstd;
    float *dmu;
    float *xc;
    float momentum = 0.9;
    float beta1 = 0.9;
    float beta2 = 0.999;
    Batchnorm(int _N);
    ~Batchnorm();
    void init();
    void print();
    void forward();
    void backward(float *delta);
    void update(update_args a);
    void save(fstream *file);
    static Batchnorm* load(char *buf);


    void get_mean();
    void get_variance();
    void normalize();
    void scale_and_shift();

    void get_mean_gpu();
    void get_variance_gpu();
    void normalize_gpu();
    void scale_and_shift_gpu();

#ifdef GPU
    void forward_gpu();
    void backward_gpu(float *delta);
#endif 
};

class Dropout : public Layer {
  public:
    int N;
    float *mask;
    float *prob;
    float ratio;
    Dropout(int _N, float _ratio);
    ~Dropout();
    void init();
    void print();
    void forward();

    void backward(float *delta);
    void update(update_args a);
    void save(fstream *file);
    static Dropout* load(char *buf);

#ifdef GPU
    void forward_gpu();
    void backward_gpu(float *delta);
#endif

};


class Shortcut : public Layer {
  public:
    int w, h, c;
    int conv_idx, actv_idx;
    Shortcut(int _w, int _h, int _c, int conv_idx, Convolution *_conv,
	     int actv_idx, Activation *_activation);
    ~Shortcut();
    float *identity;
    Convolution *conv;
    Activation *activation;
    void init();
    void print();
    void forward();
    void backward(float *delta);
    void update(update_args a);
    void save(fstream *file);
    static Shortcut* load(char *buf);

#ifdef GPU
    void forward_gpu();
    void backward_gpu(float *delta);
#endif


};


