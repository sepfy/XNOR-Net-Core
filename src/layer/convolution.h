#ifndef LAYER_CONVOLUTION_H_
#define LAYER_CONVOLUTION_H_

#include "layer.h"
#include "binary.h"

#ifdef GEMMBITSERIAL
#include "gemmbitserial.hpp"
using namespace gemmbitserial;
#endif

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

    Convolution(int W, int H, int C,
	int FW, int FH, int FC, int stride, int pad);
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
    void save(std::fstream *file);
    static Convolution* load(char *buf);

#ifdef GPU
    void forward_full_gpu();
    void forward_xnor_gpu();
    void bias_add_gpu();
    void binarize_input_gpu();
    void binarize_weight_gpu();
#endif

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

#ifdef GEMMBITSERIAL
    GEMMContext ctx;
#endif

};

#endif //  LAYER_CONVOLUTION_H_
