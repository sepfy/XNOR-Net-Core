#ifndef LAYER_SHORTCUT_H_
#define LAYER_SHORTCUT_H_

#include "layer.h"
#include "layer/activation.h"
#include "layer/convolution.h"

class Shortcut : public Layer {
  public:
    int ow, oh, oc;
    int iw, ih, ic;
    int conv_idx, actv_idx;
    Shortcut(int iw, int ih, int ic, int ow, int oh, int oc,
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
    void save(std::fstream *file);
    static Shortcut* load(char *buf);

};

#endif //  LAYER_SHORTCUT_H_
