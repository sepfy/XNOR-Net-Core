#include "layer/depthwise_conv.h"

DepthwiseConv::DepthwiseConv(
    int width,
    int height,
    int channel,
    int filter_width,
    int filter_height,
    int stride,
    int pad)
    : width_(width),
      height_(height),
      channel_(channel),
      filter_width_(filter_width),
      filter_height_(filter_height),
      stride_(stride),
      pad_(pad) {

  output_width = (width + 2*pad - filter_width)/stride + 1;
  output_height = (height + 2*pad - filter_height)/stride + 1;
  col = filter_width*filter_height;

}
