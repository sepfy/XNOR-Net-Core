#ifndef LAYER_DROPOUT_H_
#define LAYER_DROPOUT_H_

#include "layer.h"

class Dropout : public Layer {
  public:
    int N;
    float *mask;
    float *prob;
    float ratio;
    Dropout(int N, float ratio);
    ~Dropout();
    void Init();
    void Print();
    void Forward();

    void Backward(float *delta);
    void Save(std::fstream *file);
    static Dropout* load(char *buf);

};

#endif //  LAYER_DROPOUT_H_
