#include "layer.h"

__global__ void im2col_gpu_kernel(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col, int size) {

  //int k = blockIdx.x;
  //int i = threadIdx.x;
  //int j = threadIdx.y;

  int out_w = (W + 2*pad - FW)/stride + 1;
  int out_col = FH*FW*C;
  int offset_w, offset_h, c_im;
  int im_row, im_col;

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index > size) return;
  int k = index % out_col;
  int i = index / out_col / out_w;
  int j = index / out_col % out_w;


  c_im = k % C;
  offset_w = k / C % FW;
  offset_h = k / C / FW;
  
  im_row = offset_h + i*stride;
  im_col = offset_w + j*stride;

  int col_idx = (i*out_w + j)*out_col + k;
  im_row -= pad;
  im_col -= pad;

  if(im_row < 0 || im_col < 0 ||
     im_row >= H || im_col >= W)
     col[col_idx] = 0.0;
   else {
      int im_idx = C*(im_row*W + im_col) + c_im;
      col[col_idx] = im[im_idx];
   }
}

void im2col_gpu(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col) {

  int out_col = FH*FW*C;
  int out_w = (W + 2*pad - FW)/stride + 1;
  int out_h = (H + 2*pad - FH)/stride + 1;
  int size = out_col*out_w*out_h;
  //dim3 d = {(unsigned int)out_h, (unsigned int)out_w, 1};
  im2col_gpu_kernel<<<default_grid(size), BLOCK>>>(W, H, C, FW, FH, FC, stride, pad, im ,col, size);
  check_error(cudaGetLastError());
}

#if 0
__global__ void col2im_gpu_kernel(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col, int size) {

  //int k = blockIdx.x;
  //int i = threadIdx.x;
  //int j = threadIdx.y;

  int out_w = (W + 2*pad - FW)/stride + 1;
  int out_col = FH*FW*C;
  int offset_w, offset_h, c_im;
  int im_row, im_col;

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index > size) return;
  int k = index % out_col;
  int i = index / out_col / out_w;
  int j = index / out_col % out_w;

  c_im = k % C;
  offset_w = k / C % FW;
  offset_h = k / C / FW;

  im_row = offset_h + i*stride;
  im_col = offset_w + j*stride;

  int col_idx = (i*out_w + j)*out_col + k;
  im_row -= pad;
  im_col -= pad;

  if(im_row < 0 || im_col < 0 ||
     im_row >= H || im_col >= W) {}
  else {
    int im_idx = C*(im_row*W + im_col) + c_im;
    im[im_idx] = col[col_idx];
  }

}

void col2im_gpu(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col) {

  int out_col = FH*FW*C;
  int out_w = (W + 2*pad - FW)/stride + 1;
  int out_h = (H + 2*pad - FH)/stride + 1;
  //dim3 d = {(unsigned int)out_h, (unsigned int)out_w, 1};
  int size = out_col*out_w*out_h;


  col2im_gpu_kernel<<<default_grid(size), BLOCK>>>(W, H, C, FW, FH, FC, stride, pad, im ,col, size);
  check_error(cudaGetLastError());
}
#endif

#if 1
__global__ void col2im_gpu_kernel(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col, int size) {

  //int k = blockIdx.x;
  //int i = threadIdx.x;
  //int j = threadIdx.y;
  int im_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(im_idx > size) return;
  int k = im_idx % C;
  int i = im_idx / C / W;
  int j = im_idx / C % W;

  //int im_idx = (i*W + j)*C + k;

  i += pad;
  j += pad;
  int out_w_start = (j < FW) ? 0 : (j - FW)/stride + 1;
  int out_h_start = (i < FH) ? 0 : (i - FH)/stride + 1;

  int out_h = (H + 2*pad - FH)/stride + 1;
  int out_w = (W + 2*pad - FW)/stride + 1;
  int out_col = FW*FH*C;
 
  int out_h_end = i/stride + 1;
  if(out_h_end > out_h)
    out_h_end = out_h;

  int out_w_end = j/stride + 1;
  if(out_w_end > out_w)
    out_w_end = out_w;

  int h, w;
  im[im_idx] = 0.0;
  for(h = out_h_start; h < out_h_end; h++) {
    for(w = out_w_start; w < out_w_end; w++) {
      int offset_w = (j < FW) ? j : (FW - 1);
      int offset_h = (i < FH) ? i : (FH - 1);
      int col_h = (offset_h - h + out_h_start)%FH;
      int col_w = (offset_w - w + out_w_start)%FW;
      int col_idx = (h*out_w + w)*out_col + ((col_h*FW + col_w)*C) + k;
      im[im_idx] += col[col_idx];
    }
  }

}

void col2im_gpu(int W, int H, int C, int FW, int FH, int FC,
            int stride, int pad, float *im, float *col) {

  //dim3 d = {(unsigned int)H, (unsigned int)W, 1};
  int size = C*H*W;

  col2im_gpu_kernel<<<default_grid(size), BLOCK>>>(W, H, C, FW, FH, FC, stride, pad, im ,col, size);
  check_error(cudaGetLastError());
}
#endif
