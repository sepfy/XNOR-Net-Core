#ifndef LAYER_ACTIVATION_H_
#define LAYER_ACTIVATION_H_

#include "layer.h"

enum ActivationType {
  RELU,
  LEAKY,
  SIGMD,
};

class Activation : public Layer {
 public:
  Activation(int N, ActivationType activation_type) : N(N), activation_type_(activation_type) {};
  ~Activation();

  ActivationType activation_type_;
  int N;
  float *cut;
  void Init();
  void Forward();
  void Print();
  void Backward(float *delta);
  void Save(std::fstream *file);
  static Activation* load(char *buf);


 private:
  void relu_activate();
  void leaky_activate();
  void sigmoid_activate();
  void relu_backward(float *delta);
  void leaky_backward(float *delta);
  void sigmoid_backward(float *delta);

};

#endif //  LAYER_ACTIVATION_H_
