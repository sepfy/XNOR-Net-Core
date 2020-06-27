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
  Activation(int n, ActivationType activation_type) : n_(n), activation_type_(activation_type) {};
  ~Activation();

  int n_;
  ActivationType activation_type_;
  float *cut;
  void Init() override;
  void Forward() override;
  void Print() override;
  void Backward(float *delta) override;
  void Save(std::fstream *file) override;
  void LoadParams(std::fstream *rfile, int batch) override;
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
