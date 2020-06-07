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
    void Init();
    void Print();
    void Forward();
    void Backward(float *delta);
    void Save(std::fstream *file);
    static Shortcut* load(char *buf);
    void LoadParams(std::fstream *file, int batch) override;
};

#endif //  LAYER_SHORTCUT_H_
