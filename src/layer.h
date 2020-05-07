#ifndef LAYER_H_
#define LAYER_H_

#include <iostream>
#include <fstream>
#include <string.h>

#include "gemm.h"
#include "utils.h"
#include "blas.h"
#include "optimizer.h"
#ifdef GPU
#include "gpu.h"
#endif

class Layer {
 public:
  virtual void Init() = 0;
  virtual void Print() = 0;
  virtual void Forward() = 0;
  virtual void Backward(float *delta) = 0;
  virtual void Save(std::fstream *file) = 0;
  virtual void Update(UpdateArgs update_args) {};

  int batch;
  float *input;
  float *output;
  bool train_flag_ = false;
  float *delta_;
  size_t shared_size_ = 0;
  float *shared_;
  int8_t *quantized_shared_;
};

#endif //  LAYER_H_
