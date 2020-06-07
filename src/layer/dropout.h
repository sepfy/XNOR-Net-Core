#ifndef LAYER_DROPOUT_H_
#define LAYER_DROPOUT_H_

#include "layer.h"

class Dropout : public Layer {
 public:
  Dropout(int n, float ratio) : n_(n), ratio_(ratio) {}
  ~Dropout() {}
  void Init();
  void Print();
  void Forward();
  void Backward(float *delta);
  void Save(std::fstream *file);
  static Dropout* load(char *buf);
  void LoadParams(std::fstream *file, int batch) override;
 private:
  int n_;
  float ratio_;
  float *mask_;
  float *prob_;
};

#endif //  LAYER_DROPOUT_H_
