#ifndef LAYER_CONVOLUTION_H_
#define LAYER_CONVOLUTION_H_

#include "layer.h"

// Convolution layer
class Convolution : public Layer {

 public:
  Convolution(int W, int H, int C, int FW, int FH, int FC, int stride, int pad) :
  W(W), H(H), C(C), FW(FW), FH(FH), FC(FC), stride(stride), pad(pad) {

    out_w = (W + 2*pad - FW)/stride + 1;
    out_h = (H + 2*pad - FH)/stride + 1;
    out_channel = FW*FH*C;
    im_size = H*W*C;
    weight_size = out_channel*FC;
    bias_size = FC;

  }
  ~Convolution();
  void Init() override;
  void Print() override;
  void Forward() override;
  void Backward(float *delta) override;
  void Update(UpdateArgs update_args) override;
  void Save(std::fstream *file) override;
  static Convolution* load(char *buf);
  void LoadParams(std::fstream *file, int batch) override;

  int FW, FH, FC;
  int stride, pad;
  float *delta_col;
  int W, H, C;
  int out_channel;
  int out_w, out_h;
  int col_size;
  int im_size;
  int weight_size;
  int bias_size;
  int input_size;

  bool runtime = false;

  float *weight, *bias;
  float *grad_weight, *grad_bias;
  // Adam optimizer
  float *m_weight;
  float *v_weight;
  float *m_bias;
  float *v_bias;

};

#endif //  LAYER_CONVOLUTION_H_
