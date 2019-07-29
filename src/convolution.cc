#include "layers.h"
using namespace std;

Convolution::Convolution(int _W, int _H, int _C,
  int _FW, int _FH, int _FC, int _stride, bool _pad) {

  W = _W;
  H = _H;
  C = _C;
  FW = _FW;
  FH = _FH;
  FC = _FC;
  stride = _stride;


  if(_pad == true) {
    pad = 0.5*((stride - 1)*W - stride + FW);
    out_w = W;
    out_h = H;
  }
  else {
    pad = 0;
    out_w = (W - FW)/stride + 1;
    out_h = (H - FH)/stride + 1;
  }

  out_channel = FW*FH*C;
  col_size = out_w*out_h*out_channel;
  im_size = H*W*C;
  weight_size = out_channel*FC;
  bias_size = out_w*out_h*FC;
  input_size = batch*im_size;
}

Convolution::~Convolution() {

}

void Convolution::init() {

  col = new float[out_w*out_h*out_channel];
  output = new float[batch*out_w*out_h*FC];
  out_col = new float[out_w*out_h*out_channel*batch];

  weight = new float[out_channel*FC];
  grad_weight = new float[out_channel*FC];
  bias = new float[out_w*out_h*FC];
  grad_bias = new float[out_w*out_h*FC];
  im = new float[H*W*C];
  m_delta = new float[batch*W*H*C]; 

#ifdef XNOR_NET
  binary_weight = new float[out_channel*FC];
  avg_filter = new float[batch*im_size];
  avg_col = new float[out_w*out_h*FW*FC*batch];
  k_filter = new float[FW*FH];
  k_output = new float[out_w*out_h*batch];
  for(int i = 0; i < FW*FH; i++)
    k_filter[i] = 1.0/(float)(FW*FH);
  mean = new float[FC];

  bitset_outcol = new Bitset[batch*out_h*out_w];
  bitset_weight = new Bitset[FC];

  for(int i = 0; i < batch*out_h*out_w; i++)
    bitset_outcol[i].init(out_channel);

  for(int i = 0; i < FC; i++)
    bitset_weight[i].init(out_channel);

#endif

  random_normal(out_channel*FC, weight);
  random_normal(out_w*out_h*FC, bias);

  
  //Adam
  m_weight = new float[out_channel*FC];
  v_weight = new float[out_channel*FC];
  m_bias = new float[out_w*out_h*FC];
  v_bias = new float[out_w*out_h*FC];
  for(int i = 0; i < out_channel*FC; i++) {
    m_weight[i] = 0.0;
    v_weight[i] = 0.0;
  }
  for(int i = 0; i < out_w*out_h*FC; i++) {
    m_bias[i] = 0.0;
    v_bias[i] = 0.0;
  }

}

#ifdef XNOR_NET
void Convolution::swap_weight()
{
    float *swap = weight;
    weight = binary_weight;
    binary_weight = swap;
}

float Convolution::binarize_weight() {

  for(int i = 0; i < FC; i++) {
    mean[i] = 0.0;
    for(int j = 0; j < out_channel; j++) 
      mean[i] += fabs(weight[i*out_channel+j]);

    mean[i] /= (float)(out_channel);
    for(int j = 0; j < out_channel; j++) {
      int widx = i*out_channel+j;
       binary_weight[widx] = (weight[widx] > 0) ? 1.0 : -1.0;
    }
  }

}

void Convolution::binarize_input() {

  for(int b = 0; b < batch; b++) {
    for(int i = 0; i < H; i++) {
      for(int j = 0; j < W; j++) {
        int avg_idx = b*H*W + i*W + j;
        int in_idx = b*im_size + i*W*C + j*C;
        avg_filter[avg_idx] = 0.0;
        for(int k = 0; k < C; k++)
          avg_filter[avg_idx] += fabs(input[in_idx + k]);
        avg_filter[avg_idx] /= (float)C;
      }
    }
  }

  for(int i = 0; i < batch*im_size; i++) 
    input[i] > 0 ? input[i] = 1 : input[i] = -1;
 
}

#endif

void Convolution::forward() {

#ifdef XNOR_NET

  binarize_input();
  for(int i = 0; i < batch; i++)
    im2col(W, H, C, FW, FH, FC, stride, pad, 
      input + i*im_size, out_col+i*col_size);

//  ms_t start = getms();
  if(trainable) {
    binarize_weight();
    swap_weight();
    //gemm(batch*out_h*out_w, FC, out_channel, 1.0, out_col, weight, output);
    //for(int i = 0; i < batch*out_h*out_w*out_channel; i++)
    //  out_col[i] > 0 ? out_col[i] = 1 : out_col[i] = -1;
    bin_gemm(batch*out_h*out_w, FC, out_channel, 1.0, out_col, weight, output);
  }
 else {

    for(int i = 0; i < batch*out_h*out_w; i++) {
      bitset_outcol[i].clean();
      bitset_outcol[i].set(out_col+i*out_channel);
    }

    bin_gemm(batch*out_h*out_w, FC, out_channel, 1.0, 
      bitset_outcol, bitset_weight, output);
  }
//  cout << getms() - start << endl;


  // Do K = A (*) k
  for(int i = 0; i < batch; i++) 
    im2col(W, H, 1, FW, FH, 1, stride, pad, 
      avg_filter + i*W*H, avg_col + i*out_w*out_h*FW*FH);
  gemm(batch*out_h*out_w, 1, FW*FH, 1.0, avg_col, k_filter, k_output);

  for(int b = 0; b < batch; b++)
    for(int i = 0; i < out_h; i++)
      for(int j = 0; j < out_w; j++) {
        int idx = b*out_h*out_w+i*out_w+j;
        scalar(FC, k_output[idx],
         output+idx*FC, output+idx*FC);
        for(int k = 0; k < FC; k++)
          output[idx*FC + k] *= mean[k];
      }

#else
  for(int i = 0; i < batch; i++)
    im2col(W, H, C, FW, FH, FC, stride, pad, 
      input + i*im_size, out_col+i*col_size);
  gemm(batch*out_h*out_w, FC, out_channel, 1, out_col, weight, output);
#endif

  bias_add(batch, out_h*out_w*FC, output, bias);

#ifdef XNOR_NET
  swap_weight();
#endif

}

void Convolution::backward(float *delta) {

  //weight
  memset(grad_weight, 0, out_channel*FC*sizeof(float));
  gemm_ta(out_channel, FC, out_h*out_w*batch, 1.0, out_col, delta, grad_weight);

  float *tmp = new float[out_channel*FC];
  scalar(out_channel*FC, 0.01/(float)(out_channel*FC), weight, tmp);
  add(out_channel, FC, grad_weight, tmp, grad_weight);
  delete[] tmp;


  //bias
  memset(grad_bias, 0, out_w*out_h*FC*sizeof(float));
  row_sum(batch, out_w*out_h*FC, delta, grad_bias);

  float *delta_col = new float[batch*out_channel*out_w*out_h];
  gemm_tb(batch*out_w*out_h, out_channel, FC, 1.0, delta, weight, delta_col);



  for(int i = 0; i < batch; i++)
    col2im(W,H, C, FW, FH, FC, stride, pad, 
      m_delta + i*im_size, delta_col + i*col_size); 

  delete[] delta_col;

}

void Convolution::update(float lr) {

  //Adam optimizer

#if 1
  iter++;
  float m_lr = lr * pow(1.0 - pow(beta2, iter), 0.5) / (1.0 - pow(beta1, iter));
  for(int i = 0; i < out_channel*FC; i++) {
    m_weight[i] = (1 - beta1)*grad_weight[i] + beta1*m_weight[i];
    v_weight[i] = (1 - beta2)*pow(grad_weight[i], 2.0) + beta2*v_weight[i];
  }

  for(int i = 0; i < out_channel*FC; i++) {
    weight[i] -= m_lr * m_weight[i]/(pow(v_weight[i], 0.5) + eplson);
  }  

  for(int i = 0; i < out_w*out_h*FC; i++) {
    m_bias[i] = (1 - beta1)*grad_bias[i] + beta1*m_bias[i];
    v_bias[i] = (1 - beta2)*pow(grad_bias[i], 2.0) + beta2*v_bias[i];
  }

  for(int i = 0; i < out_w*out_h*FC; i++) {
    bias[i] -= m_lr * m_bias[i]/(pow(v_bias[i], 0.5) + eplson);
  }
#endif  


#if 0
  mat_scalar(out_channel, FC, grad_weight, lr, grad_weight);
  mat_minus(out_channel, FC, weight, grad_weight, weight);
  mat_scalar(1, out_w*out_h*FC, grad_bias, lr, grad_bias);
  mat_minus(1, out_w*out_h*FC, bias, grad_bias, bias);
#endif

}

void Convolution::save(fstream *file) {

  char buf[64] = {0};
  sprintf(buf, "Convolution,%d,%d,%d,%d,%d,%d,%d,%d", 
    W, H, C, FW, FH, FC, stride, pad);
  //cout << weight[0] << endl;
  //cout << bias[0] << endl;
  file->write(buf, sizeof(buf));
#ifdef XNOR_NET
  float *BB = new float[FC*out_channel];
  for(int i = 0; i < FC; i++)
    for(int j = 0; j < out_channel; j++)
      BB[i*out_channel+j] = weight[j*FC+i];

  for(int i = 0; i < FC; i++) {
    bitset_weight[i].set(BB+i*out_channel);
  }
  delete[] BB;

  for(int i = 0; i < FC; i++) {
    file->write((char*)bitset_weight[i].bits,
                       bitset_weight[i].N*sizeof(uint64_t));
  }

  file->write((char*)mean, FC*sizeof(float));
 
#else
  file->write((char*)weight, weight_size*sizeof(float));
#endif
  file->write((char*)bias, bias_size*sizeof(float));


}

Convolution* Convolution::load(char *buf) {

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

  Convolution *conv = new Convolution(para[0], para[1], 
  para[2], para[3], para[4], para[5], para[6], para[7]);
  return conv;
}
