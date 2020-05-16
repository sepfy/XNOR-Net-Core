#ifndef LAYER_BINARY_CONV_H_
#define LAYER_BINARY_CONV_H_

#include "layer.h"
#include "binary.h"

#ifdef GEMMBITSERIAL
#include "gemmbitserial.hpp"
using namespace gemmbitserial;
#endif

class BinaryConv : public Layer {

 public:
  BinaryConv(
    int width,
    int height,
    int channel,
    int filter_width,
    int filter_height,
    int filter_channel,
    int stride,
    int pad);

  ~BinaryConv() {};

  void Init() override;
  void Print() override;
  void Forward() override;
  void Backward(float *delta) override;
  void Update(UpdateArgs update_args) override;
  void Save(std::fstream *file) override;
  static BinaryConv* load(char *buf);

// private:
  int width, height, channel;
  int filter_width, filter_height, filter_channel, filter_col;
  int stride, pad;
  int out_w, out_h;

  int col_size;
  int im_size;
  int weight_size;
  int bias_size;
  int input_size;



  bool runtime = false;

  float *mean;
  float *binary_weight;
  float *binary_input;
  float *avg_filter;
  float *avg_col;
  float *k_filter;
  float *k_output;
  Bitset *bitset_outcol, *bitset_weight;

  void SwapWeight();
  void BinarizeWeight();
  void BinarizeInput();
  void BiasAdd();


  float *weight, *bias;
  float *grad_weight, *grad_bias;
  // Adam optimizer
  float *m_weight;
  float *v_weight;
  float *m_bias;
  float *v_bias;

   
#ifdef GEMMBITSERIAL
  GEMMContext ctx;
#endif

 

}; 

#endif //  LAYER_BINARY_CONV_H_
