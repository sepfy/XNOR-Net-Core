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


void BinaryConv::Save(std::fstream *file) {

  char buf[64] = {0};
  sprintf(buf, "BinaryWeight,%d,%d,%d,%d,%d,%d,%d,%d",
    width, height, channel, filter_width, filter_height, filter_channel, stride, pad);
  file->write(buf, sizeof(buf));

#ifdef GEMMBITSERIAL

  int8_t *BB = new int8_t[filter_col*filter_channel];
#ifdef GPU
  float *weight_tmp = new float[weight_size];
  gpu_pull_array(weight, weight_tmp, weight_size);
  for(int i = 0; i < FC; i++)
    for(int j = 0; j < out_channel; j++)
      BB[i*filter_col+j] = (weight_tmp[j*filter_channel+i] > 0) ? 1 : -1;
  delete []weight_tmp;
#endif
  ctx.rhs.importRegular(BB);
  size_t size = ctx.rhs.nbits*ctx.rhs.wordsPerBitplane()*sizeof(uint64_t);
  //cout << FC*out_channel << endl;
  file->write((char*)ctx.rhs.data, size);

#else
    float *BB = new float[filter_channel*filter_col];
#ifdef GPU
    float *weight_tmp = new float[weight_size];
    gpu_pull_array(weight, weight_tmp, weight_size);
    for(int i = 0; i < filter_channel; i++)
      for(int j = 0; j < filter_col; j++)
        BB[i*filter_col+j] = weight_tmp[j*filter_channel+i];
    delete []weight_tmp;
#else
    for(int i = 0; i < filter_channel; i++)
      for(int j = 0; j < filter_col; j++)
        BB[i*filter_col+j] = weight[j*filter_channel+i];
#endif
    for(int i = 0; i < filter_channel; i++) {
      bitset_weight[i].set(BB+i*filter_col);
    }
    delete[] BB;

    for(int i = 0; i < filter_channel; i++) {
      file->write((char*)bitset_weight[i].bits,
                         bitset_weight[i].N*sizeof(BIT_BLK));
    }
#endif

#ifdef GPU
    BinarizeWeight();
    float *mean_tmp = new float[filter_channel];
    gpu_pull_array(mean, mean_tmp, filter_channel);
    file->write((char*)mean_tmp, filter_channel*sizeof(float));
    delete []mean_tmp;
#else
    BinarizeWeight();
    file->write((char*)mean, filter_channel*sizeof(float));
#endif
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

