#ifndef LAYER_SOFTMAX_H_
#define LAYER_SOFTMAX_H_

#include "layer.h"

class SoftmaxWithCrossEntropy : public Layer {
 public:
  SoftmaxWithCrossEntropy(int n) : n_(n) {}
  ~SoftmaxWithCrossEntropy() {}
  void Init();
  void Print();
  void Forward();
  void Backward(float *delta);
  void Save(std::fstream *file);
  static SoftmaxWithCrossEntropy* load(char *buf);
  void LoadParams(std::fstream *file, int batch) override;
 private:
  int n_;

};

#endif //  LAYER_SOFTMAX_H_
