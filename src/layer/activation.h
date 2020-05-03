#ifndef LAYER_ACTIVATION_H_
#define LAYER_ACTIVATION_H_

#include "layer.h"

enum ACT{
  RELU,
  LEAKY,
  SIGMD,
  NUM_TYPE
};

class Activation : public Layer {
  public:
    
    ACT activation;
    void relu_activate();
    void leaky_activate();
    void sigmoid_activate();
    void relu_backward(float *delta);
    void leaky_backward(float *delta);
    void sigmoid_backward(float *delta);

    int N;
    float *cut;
    Activation(int N, ACT act);
    ~Activation();
    void init();
    void print();
    void forward();
    void backward(float *delta);
    void update(update_args a);
    void save(std::fstream *file);
    static Activation* load(char *buf);

#ifdef GPU
    void relu_activate_gpu();
    void leaky_activate_gpu();
    void relu_backward_gpu(float *delta);
    void leaky_backward_gpu(float *delta);
#endif
};

#endif //  LAYER_ACTIVATION_H_
