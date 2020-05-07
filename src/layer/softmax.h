#ifndef LAYER_SOFTMAX_H_
#define LAYER_SOFTMAX_H_

#include "layer.h"

class SoftmaxWithCrossEntropy : public Layer {

  public:
    int N;

    SoftmaxWithCrossEntropy(int n);
    ~SoftmaxWithCrossEntropy();
    void Init();
    void Print();
    void Forward();
    void Backward(float *delta);
    void Save(std::fstream *file);
    static SoftmaxWithCrossEntropy* load(char *buf);

};

#endif //  LAYER_SOFTMAX_H_
