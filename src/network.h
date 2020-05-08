#include <vector>
#include <map>
#include <fstream>
#include <string>

#include "layer.h"
#include "layer/batchnorm.h"
#include "layer/connected.h"
#include "layer/convolution.h"
#include "layer/activation.h"
#include "layer/maxpool.h"
#include "layer/avgpool.h"
#include "layer/dropout.h"
#include "layer/shortcut.h"
#include "layer/softmax.h"
#include "gemm.h"
#include "blas.h"
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include <string.h>
#include <fstream>
#include "binary.h"
#include "optimizer.h"

class Network {
 public:
  Network() {};
  ~Network() {};
  inline void Add(Layer* layer) { layers_.push_back(layer); }
  void Init(int batch, float lr, bool use_adam);
  void Save(char *filename);
  void Load(char *filename, int batch);
  void Deploy();
  void Train(float *correct);
  void Inference(float *input);
  float* output() { return output_; }

 private:
  vector<Layer*> layers_;
  UpdateArgs update_args_;
  int8_t *quantized_shared_;
  float *shared_;
  float *output_;
};
