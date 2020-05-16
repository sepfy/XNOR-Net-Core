#include "layer/convolution.h"
#include "utils.h"
#include "blas.h"
#include "gemm.h"


void Convolution::Print() {

  if(xnor)
    printf("ConvolutionX \t %d x %d x %d \t\t %d x %d x %d \n", H, W, C, out_h, out_w, FC);
  else 
    printf("Convolution \t %d x %d x %d \t\t %d x %d x %d \n", H, W, C, out_h, out_w, FC);

}

void Convolution::swap_weight() {
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

  if(!runtime) {
    for(int i = 0; i < batch*out_w*out_h*out_channel; i++)
      shared_[i] > 0 ? shared_[i] = 1 : shared_[i] = -1;
  }
/*  else {
    for(int i = 0; i < batch*out_w*out_h*out_channel; i++)
      quantized_shared_[i] = (shared_[i] > 0) ? 1 : -1;
  }
 */
}

void Convolution::forward_xnor() {

  if(!runtime) {

    for(int i = 0; i < batch; i++)
      im2col(W, H, C, FW, FH, FC, stride, pad,
        input + i*im_size, shared_+i*col_size);

    binarize_input();

    binarize_weight();
    swap_weight();
    gemm_cpu(TRS_N, TRS_N,
             batch*out_h*out_w, FC, out_channel, 1.0,
              shared_, weight, output);
  }
  else {

    for(int i = 0; i < batch; i++)
      im2col_xnor(W, H, C, FW, FH, FC, stride, pad,
        input + i*im_size, quantized_shared_+i*col_size);

    binarize_input();
#ifdef GEMMBITSERIAL
    ctx.lhs.importRegular(quantized_shared_);
    gemmBitSerial(ctx);

    // Transpose
    for(int i = 0; i < batch*out_w*out_h; i++)
      for(int j = 0; j < FC; j++)
        output[i*FC+j] = ctx.res[j*(batch*out_w*out_h)+i];
#else
    for(int i = 0; i < batch*out_h*out_w; i++) {
      bitset_outcol[i].clean();
      bitset_outcol[i].set(shared_+i*out_channel);
    }
  //ms_t s = getms();
    bin_gemm(batch*out_h*out_w, FC, out_channel, 1.0, 
      bitset_outcol, bitset_weight, output);
  //cout << "bin_gemm: " << getms() -s << endl;
#endif
  }

  // Do K = A (*) k
  for(int i = 0; i < batch; i++) 
    im2col(W, H, 1, FW, FH, 1, stride, pad, 
      avg_filter + i*W*H, avg_col + i*out_w*out_h*FW*FH);
  gemm_cpu(TRS_N, TRS_N, batch*out_h*out_w, 1, FW*FH, 1.0, avg_col, k_filter, k_output);

  for(int b = 0; b < batch; b++)
    for(int i = 0; i < out_h; i++)
      for(int j = 0; j < out_w; j++) {
        int idx = b*out_h*out_w+i*out_w+j;
        scalar(FC, k_output[idx],
         output+idx*FC, output+idx*FC);
        for(int k = 0; k < FC; k++)
          output[idx*FC + k] *= mean[k];
      }

  if(xnor && !runtime) {
    swap_weight();
  }
}

void Convolution::forward_full() {

  for(int i = 0; i < batch; i++)
    im2col(W, H, C, FW, FH, FC, stride, pad, 
      input + i*im_size, shared_+i*col_size);

  gemm_cpu(TRS_N, TRS_N, batch*out_h*out_w, FC, out_channel, 1, shared_, weight, output);

}

#ifndef GPU


void Convolution::Init() {

  output = new float[batch*out_w*out_h*FC];
  weight = new float[out_channel*FC];
  bias = new float[FC];

  // XNOR  
  binary_weight = new float[out_channel*FC];
  avg_filter = new float[batch*im_size];
  avg_col = new float[out_w*out_h*FW*FC*batch];
  k_filter = new float[FW*FH];
  k_output = new float[out_w*out_h*batch];
  for(int i = 0; i < FW*FH; i++)
    k_filter[i] = 1.0/(float)(FW*FH);
  mean = new float[FC];

  if(train_flag_) {
    grad_weight = new float[out_channel*FC];
    grad_bias = new float[FC];
    delta_ = new float[batch*W*H*C]; 

    /* Adam optimizer */
    m_weight = new float[out_channel*FC];
    v_weight = new float[out_channel*FC];
    m_bias = new float[FC];
    v_bias = new float[FC];

    random_normal(out_channel*FC, weight);
    random_normal(FC, bias);
    memset(m_weight, 0, out_channel*FC*sizeof(float));
    memset(v_weight, 0, out_channel*FC*sizeof(float));
    memset(m_bias, 0 , FC*sizeof(float));  
    memset(v_bias, 0 , FC*sizeof(float));  
  }



#ifdef GEMMBITSERIAL
  ctx = allocGEMMContext(batch*out_h*out_w, out_channel, FC, 1, 1, 1, 1);
#else
  bitset_outcol = new Bitset[batch*out_h*out_w];
  bitset_weight = new Bitset[FC];

  for(int i = 0; i < batch*out_h*out_w; i++)
    bitset_outcol[i].Init(out_channel);

  for(int i = 0; i < FC; i++)
    bitset_weight[i].Init(out_channel);
#endif

  shared_size_ = out_w*out_h*out_channel*batch;
  input_size = batch*im_size;
}



void Convolution::Forward() {

  if(xnor)
    forward_xnor();
  else 
    forward_full();

  bias_add();
}



void Convolution::Backward(float *delta) {

  for(int i = 0; i < batch; i++)
    im2col(W, H, C, FW, FH, FC, stride, pad,
      input + i*im_size, shared_+i*col_size);

  gemm_cpu(TRS_T, TRS_N, 
           out_channel, FC, out_h*out_w*batch, 1.0,
           shared_, delta, grad_weight);

  if(xnor) {
/*
    for(int i = 0; i < out_channel; i++)
      for(int j = 0; j < FC; j++) {
        int idx = i*FC+j;
        grad_weight[idx] = (grad_weight[idx]*(1.0/(float)(out_channel) 
                         + mean[j]*(fabs(weight[idx]) <= 1 ? weight[idx] : 0)))*(1.0 - 1.0/(float)C)*out_channel;
      }
*/
    //TODO: binary_weight? or weight
    gemm_cpu(TRS_N, TRS_T,
           batch*out_w*out_h, out_channel, FC, 1.0,
           delta, weight, shared_);

  }
  else {
    gemm_cpu(TRS_N, TRS_T,
           batch*out_w*out_h, out_channel, FC, 1.0,
           delta, weight, shared_);
  }

  row_sum(batch*out_w*out_h, FC, delta, grad_bias);

  for(int i = 0; i < batch; i++)
    col2im(W,H, C, FW, FH, FC, stride, pad, 
      delta_ + i*im_size, shared_ + i*col_size);

}

void Convolution::bias_add() {
  for(int b = 0; b < batch; b++)
    for(int i = 0; i < out_w*out_h; i++)
        for(int j = 0; j < FC; j++)
          output[b*out_w*out_h*FC + i*FC + j] += bias[j];
}

void Convolution::Update(UpdateArgs update_args) {

  adam_cpu(out_channel*FC, weight, grad_weight, m_weight, v_weight, update_args);
  adam_cpu(FC, bias, grad_bias, m_bias, v_bias, update_args);

}
#endif

void Convolution::Save(std::fstream *file) {

  char buf[64] = {0};
  sprintf(buf, "Convolution,%d,%d,%d,%d,%d,%d,%d,%d,%d", 
    W, H, C, FW, FH, FC, stride, pad, xnor);
  //cout << weight[0] << endl;
  //cout << bias[0] << endl;
  file->write(buf, sizeof(buf));
/*
  if(xnor) {

#ifdef GEMMBITSERIAL

  int8_t *BB = new int8_t[FC*out_channel];
#ifdef GPU
  float *weight_tmp = new float[weight_size];
  gpu_pull_array(weight, weight_tmp, weight_size);
  for(int i = 0; i < FC; i++)
    for(int j = 0; j < out_channel; j++)
      BB[i*out_channel+j] = (weight_tmp[j*FC+i] > 0) ? 1 : -1;
  delete []weight_tmp;
#endif
  ctx.rhs.importRegular(BB);
  size_t size = ctx.rhs.nbits*ctx.rhs.wordsPerBitplane()*sizeof(uint64_t);
  //cout << FC*out_channel << endl;
  file->write((char*)ctx.rhs.data, size);

#else
    float *BB = new float[FC*out_channel];
#ifdef GPU
    float *weight_tmp = new float[weight_size];
    gpu_pull_array(weight, weight_tmp, weight_size);
    for(int i = 0; i < FC; i++)
      for(int j = 0; j < out_channel; j++)
        BB[i*out_channel+j] = weight_tmp[j*FC+i];
    delete []weight_tmp;
#else
    for(int i = 0; i < FC; i++)
      for(int j = 0; j < out_channel; j++)
        BB[i*out_channel+j] = weight[j*FC+i];
#endif
    for(int i = 0; i < FC; i++) {
      bitset_weight[i].set(BB+i*out_channel);
    }
    delete[] BB;

    for(int i = 0; i < FC; i++) {
      file->write((char*)bitset_weight[i].bits,
                         bitset_weight[i].N*sizeof(BIT_BLK));
    } 
#endif

#ifdef GPU
    binarize_weight_gpu();
    float *mean_tmp = new float[FC];
    gpu_pull_array(mean, mean_tmp, FC);
    file->write((char*)mean_tmp, FC*sizeof(float));
    delete []mean_tmp;
#else
    binarize_weight();
    file->write((char*)mean, FC*sizeof(float));
#endif
  } 
  else {
*/
#ifdef GPU
    float *weight_tmp = new float[weight_size];
    gpu_pull_array(weight, weight_tmp, weight_size);
    file->write((char*)weight_tmp, weight_size*sizeof(float));
    delete []weight_tmp;
#else
    file->write((char*)weight, weight_size*sizeof(float));
#endif

//  }



#ifdef GPU
  float *bias_tmp = new float[bias_size];
  gpu_pull_array(bias, bias_tmp, bias_size);
  file->write((char*)bias_tmp, bias_size*sizeof(float));
  delete []bias_tmp;
#else  
  file->write((char*)bias, bias_size*sizeof(float));
#endif

}

Convolution* Convolution::load(char *buf) {

  int para[9] = {0};
  int idx = 0;

  char *token;
  while (buf) {
    token = strtok(NULL, ",");
    para[idx] = atoi(token);
    idx++;
    if(idx > 8)
      break;
  }

  Convolution *conv = new Convolution(para[0], para[1], 
  para[2], para[3], para[4], para[5], para[6], para[7]);
  conv->xnor = para[8];
  return conv;
}
