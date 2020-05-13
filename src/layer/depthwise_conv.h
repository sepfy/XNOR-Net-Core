#ifndef LAYER_DEPTHWISE_CONV_H_
#define LAYER_DEPTHWISE_CONV_H_

#include "layer.h"

// Depthwise Convolution layer
class DepthwiseConv : public Layer {

 public:
  DepthwiseConv(
    int width, 
    int height,
    int channel, 
    int filter_width,
    int filter_height,
    int stride,
    int pad);
  ~DepthwiseConv();
  void Init() override;
  void Print() override;
  void Forward() override;
  void Backward(float *delta) override;
  void Update(UpdateArgs update_args) override;
  void Save(std::fstream *file) override;
  static DepthwiseConv* load(char *buf);


 private:
  int width_;
  int height_;
  int channel_;
  int filter_width_;
  int filter_height_;
  int stride_;
  int pad_;
  int output_width;
  int output_height;
  int col;
};

#endif //  LAYER_DEPTHWISE_CONV_H_
