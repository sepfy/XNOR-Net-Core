#ifndef LAYER_SOFTMAX_H_
#define LAYER_SOFTMAX_H_

#include "layer.h"

class SoftmaxWithCrossEntropy : public Layer {

  public:
    int N;

    SoftmaxWithCrossEntropy(int n);
    ~SoftmaxWithCrossEntropy();
    void init();
    void print();
    void forward();
    void backward(float *delta);
    void update(update_args a);
    void save(std::fstream *file);
    static SoftmaxWithCrossEntropy* load(char *buf);

};

#endif //  LAYER_SOFTMAX_H_
