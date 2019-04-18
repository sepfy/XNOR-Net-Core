#include <iostream>
#include "utils.h"

using namespace std;

void im2col(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col) {

  int out_w = (W + 2*pad - FW)/stride + 1;
  int out_h = (H + 2*pad - FH)/stride + 1;
  int out_col = FH*FW*C;
  int offset_w, offset_h, c_im;
  int im_row, im_col;
  
  for(int i = 0; i < out_h; i++) 
    for(int j = 0; j < out_w; j++) {
      for(int k = 0; k < out_col; k++) {
        c_im = k % C;
        offset_w = k / C % FW;
        offset_h = k / C / FW;
        //c_im = k / (FH*FW);
        //offset_w = k % FW;
        //offset_h = k / FW % FH;
        im_row = offset_h + i*stride;
        im_col = offset_w + j*stride;
        //int im_idx = C*(im_row*W + im_col) + c_im;
        int im_idx = C*(im_row*W + im_col) + c_im;
        col[(i*out_w + j)*out_col + k] = im[im_idx];
        //cout << col[(i*out_w + j)*out_col + k] << ", ";
      }
    }
}
/*
void col2im(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col) {

  int out_w = (W + 2*pad - FW)/stride + 1;
  int out_h = (H + 2*pad - FH)/stride + 1;
  int out_col = FH*FW*C;
  int offset_w, offset_h, c_im;
  int im_row, im_col;

  for(int i = 0; i < out_h; i++) 
    for(int j = 0; j < out_w; j++) {
      for(int k = 0; k < out_col; k++) {
        c_im = k % C;
        offset_w = k % C / FW;
        offset_h = k / C / FW;
        im_row = offset_h + i*stride;
        im_col = offset_w + j*stride;
        int im_idx = C*(im_row*W + im_col) + c_im;
        im[m_idx] = col[(i*out_w + j)*out_col + k];
      }
}
*/
