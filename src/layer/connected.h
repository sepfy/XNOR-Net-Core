#ifndef LAYER_CONNECTED_H_
#define LAYER_CONNECTED_H_

#include "layer.h"

// Fully connected layer
// From n neurons to m neurons. So weight is a n*m matrix.
class Connected : public Layer {
 public:
  Connected(int n, int m) : n_(n), m_(m) {};
  ~Connected();
  void Init() override; 
  void Print() override;
  void Forward() override;
  void Backward(float *delta) override;
  void Update(UpdateArgs update_args) override;
  void Save(std::fstream *file) override;
  static Connected* load(char *buf);

  int n_;
  int m_;
  float *weight;
  float *bias;
  float *grad_weight;
  float *grad_bias;

 private:
  void bias_add();
  // The following variables are for adam optimzer.
  float *m_weight;
  float *v_weight;
  float *m_bias;
  float *v_bias;
};

#endif //  LAYER_CONNECTED_H_
