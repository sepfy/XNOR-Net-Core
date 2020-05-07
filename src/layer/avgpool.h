#ifndef LAYER_AVGPOOL_H_
#define LAYER_AVGPOOL_H_

#include "layer.h"

class Avgpool : public Layer {

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
    Avgpool(int W, int H, int C,
	int FW, int FH, int FC, int stride, bool pad);
    ~Avgpool();
    void Init();
    void Print();
    void Forward();
    void Backward(float *delta);
    void Save(std::fstream *file);
    static Avgpool* load(char *buf);
};

#endif //  LAYER_AVGPOOL_H_
