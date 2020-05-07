#ifndef LAYER_MAXPOOL_H_
#define LAYER_MAXPOOL_H_

#include "layer.h"

class Maxpool : public Layer {
 public:
  Maxpool(int w, int h, int c, int fw, int fh, int fc, int stride, bool pad)
    : W(w), H(h), C(c), FW(fw), FH(fw), FC(fc), stride(stride) {

  if(pad == true) {
    this->pad = 0.5*((stride - 1)*W - stride + FW);
    out_w = W;
    out_h = H;
  }
  else {
    this->pad = 0;
    out_w = (W - FW)/stride + 1;
    out_h = (H - FH)/stride + 1;
  }

  out_channel = FW*FH*C;
  }


  ~Maxpool() {};
  void Init();
  void Print();
  void Forward();
  void Backward(float *delta);
  void Save(std::fstream *file);
  static Maxpool* load(char *buf);


  float *col;
  int FW, FH, FC;
  int stride, pad;
  int W, H, C;
  int out_channel;
  int out_w, out_h;
  float *weight, *bias, *out_col, *im;
  float *grad_weight;
  float *delta_col;
  float *indexes;

};

#endif //  LAYER_MAXPOOL_H_
