#include "binary_conv.h"

BinaryConv::BinaryConv(
    int width,
    int height,
    int channel,
    int filter_width,
    int filter_height,
    int filter_channel,
    int stride,
    int pad)
    : width(width),
    height(height),
    channel(channel),
    filter_width(filter_width),
    filter_height(filter_height),
    filter_channel(filter_channel),
    stride(stride),
    pad(pad) {
  
  out_w = (width + 2*pad - filter_width)/stride + 1;
  out_h = (height + 2*pad - filter_height)/stride + 1;
  filter_col = filter_width*filter_height*channel;
  col_size = out_w*out_h*filter_col;
  im_size = width*height*channel;
  weight_size = filter_col*filter_channel;
  bias_size = filter_channel;
 
}

void BinaryConv::Print() {

  printf("BinaryConv \t %d x %d x %d \t\t %d x %d x %d \n",
      height,
      width,
      channel,
      out_h,
      out_w,
      filter_channel);
}

BinaryConv* BinaryConv::load(char *buf) {

  int para[8] = {0};
  int idx = 0;

  char *token;
  while (buf) {
    token = strtok(NULL, ",");
    para[idx] = atoi(token);
    idx++;
    if(idx > 7)
      break;
  }

  BinaryConv *conv = new BinaryConv(para[0], para[1],
  para[2], para[3], para[4], para[5], para[6], para[7]);
  return conv;
}


#ifndef GPU
void BinaryConv::Init() {

  output = new float[batch*out_w*out_h*filter_channel];
  bias = new float[filter_channel];
  mean = new float[filter_channel];

#ifdef GEMMBITSERIAL
  ctx = allocGEMMContext(batch*out_h*out_w, filter_col, filter_channel, 1, 1, 1, 1);
#else
  bitset_outcol = new Bitset[batch*out_h*out_w];
  bitset_weight = new Bitset[filter_channel];

  for(int i = 0; i < batch*out_h*out_w; i++)
    bitset_outcol[i].Init(filter_col);

  for(int i = 0; i < filter_channel; i++)
    bitset_weight[i].Init(filter_col);
#endif

  shared_size_ = out_w*out_h*filter_col*batch;
  input_size = batch*im_size;

}

void BinaryConv::BinActive() {

  size_t n = batch*out_w*out_h*filter_col;
  for(int i = 0; i < n; i++)
    shared_[i] = (shared_[i] > 0) ? 1.0 : -1.0;

}

void BinaryConv::BiasAdd() {


}

void BinaryConv::Forward() {

  for(int i = 0; i < batch; i++) {
    im2col(
        width,
        height,
        channel,
        filter_width,
        filter_height,
        filter_channel,
        stride,
        pad,
        input + i*im_size,
        shared_ + i*col_size);
  }
      
  BinActive();
  bias_add(output, bias, batch*out_w*out_h, channel);
}

void BinaryConv::Backward(float *delta) {
  // Do not support CPU training.
}

void BinaryConv::Update(UpdateArgs update_args) {
  // Do not support CPU training.
}

void BinaryConv::Save(std::fstream *file) {
  // Do not support CPU training.
}

#endif
