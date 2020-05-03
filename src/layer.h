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
  virtual void forward() = 0;
  virtual void backward(float* delta) = 0;
  virtual void update(update_args a) = 0;
  virtual void init() = 0;
  virtual void print() = 0;
  virtual void save(std::fstream *file) = 0;

  int batch;
  bool train_flag = false;
  float *input;
  float *output;
  float *m_delta;
  size_t shared_size = 0;
  float *shared;
  int8_t *quantized_shared;
};

#endif //  LAYER_H_
