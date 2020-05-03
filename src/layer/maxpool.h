#ifndef LAYER_MAXPOOL_H_
#define LAYER_MAXPOOL_H_

#include "layer.h"

class MaxPool : public Layer {
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
    MaxPool(int W, int H, int C,
	int FW, int FH, int FC, int stride, bool pad);
    ~MaxPool();
    void init();
    void print();
    void forward();
    void backward(float *delta);
    void update_gpu(update_args a);
    void update(update_args a);
    void save(std::fstream *file);
    static MaxPool* load(char *buf);

};

#endif //  LAYER_MAXPOOL_H_
