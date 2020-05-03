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


using namespace std; 

class Network {
  public:
    vector<Layer*> layers;
    float lr;
    float *shared;
    int8_t *quantized_shared;
    update_args a;
    Network();
    ~Network();
    void add(Layer* layer);
    void initial(int batch, float _lr, bool use_adam);
    float* inference(float *input);
    void train(float *Y);
    void save(char *filename);
    void load(char *filename, int batch);
    void deploy();
};
