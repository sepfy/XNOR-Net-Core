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
    void init();
    void print();
    void forward();

    void backward(float *delta);
    void update(update_args a);
    void save(std::fstream *file);
    static Dropout* load(char *buf);

};

#endif //  LAYER_DROPOUT_H_
