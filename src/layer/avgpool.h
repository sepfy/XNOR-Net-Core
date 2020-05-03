#ifndef LAYER_AVGPOOL_H_
#define LAYER_AVGPOOL_H_

#include "layer.h"

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
    AvgPool(int W, int H, int C,
	int FW, int FH, int FC, int stride, bool pad);
    ~AvgPool();
    void init();
    void print();
    void forward();
    void backward(float *delta);
    void update(update_args a);
    void save(std::fstream *file);
    static AvgPool* load(char *buf);
};

#endif //  LAYER_AVGPOOL_H_
