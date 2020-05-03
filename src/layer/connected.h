#ifndef LAYER_CONNECTED_H_
#define LAYER_CONNECTED_H_

#include "layer.h"

class Connected : public Layer {
 public:
  float *weight;
  float *bias;
  float *grad_weight;
  float *grad_bias;
  // W is NxM matrix
  int N;   
  int M;
  
  // Adam optimizer
  float beta1 = 0.9;
  float beta2 = 0.999;
  float *m_weight;
  float *v_weight;
  float *m_bias;
  float *v_bias;
  float iter = 0.0;
  float epsilon = 1.0e-7;
 
  Connected(int n, int m);
  ~Connected();
  void init(); 
  void print();
  void forward();
  void bias_add();
  void backward(float *delta);
  void update(update_args a);
  void save(std::fstream *file);
  static Connected* load(char *buf);

};

#endif //  LAYER_CONNECTED_H_
