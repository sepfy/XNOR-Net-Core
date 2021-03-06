#include <iostream>
#include "utils.h"
#include <sys/time.h>
#include <random>
using namespace std;

void im2col(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col) {

  int out_w = (W + 2*pad - FW)/stride + 1;
  int out_h = (H + 2*pad - FH)/stride + 1;
  int out_col = FH*FW*C;
  int offset_w, offset_h, c_im;
  int im_row, im_col;

  for(int k = 0; k < out_col; k++) {
    c_im = k % C;
    offset_w = k / C % FW;
    offset_h = k / C / FW;

    for(int i = 0; i < out_h; i++) {
      for(int j = 0; j < out_w; j++) {

        im_row = offset_h + i*stride;
        im_col = offset_w + j*stride;
        

        int col_idx = (i*out_w + j)*out_col + k;
        int im_pad_row = im_row - pad;
        int im_pad_col = im_col - pad;

        if(im_pad_row < 0 || im_pad_col < 0 ||
           im_pad_row >= H || im_pad_col >= W)
          col[col_idx] = 0.0;
        else { 
          int im_idx = C*(im_row*W + im_col) + c_im;
          col[col_idx] = im[im_idx];
        }
      }
    }
  }

}

void col2im(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col) {

  int out_w = (W + 2*pad - FW)/stride + 1;
  int out_h = (H + 2*pad - FH)/stride + 1;
  int out_col = FH*FW*C;
  int offset_w, offset_h, c_im;
  int im_row, im_col;
 

 
  for(int k = 0; k < out_col; k++) {

    c_im = k % C;
    offset_w = k / C % FW;
    offset_h = k / C / FW;

    for(int i = 0; i < out_h; i++) { 
      for(int j = 0; j < out_w; j++) {
        im_row = offset_h + i*stride;
        im_col = offset_w + j*stride;
        int im_idx = C*(im_row*W + im_col) + c_im;
        int col_idx = (i*out_w + j)*out_col + k;
        im[im_idx] = col[col_idx];
      }
    }
  }

}



float argmax(int N, float *data) {


}


unsigned long long getms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec*1.0e+3 + tv.tv_usec/1000;
}

void random_normal(int size, float *mat) {

  default_random_engine generator = default_random_engine(time(NULL));
  normal_distribution<float> distribution(0, 0.1);
  int i;
  for(i = 0; i < size; i++)
    mat[i] = distribution(generator);
}
