#include "layer/shortcut.h"


__global__ void shortcut_forward_gpu_kernel(float *input, float *output, float *identity, int size,
  int iw, int ih, int ic, int ow, int oh, int oc) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index >= size)
    return;

  int minw = (ow < iw) ? ow : iw;
  int minh = (oh < ih) ? oh : ih;
  int maxc = (oc > ic) ? oc : ic;

  int k = index%maxc;
  int j = index/maxc%minw;
  int i = index/maxc/minw%minh;
  int b = index/maxc/minw/minh;

  int sample = iw/ow;
  int stride = ic/oc;


  int out_idx = b*oh*ow*oc + i*ow*oc + j*oc + k;
  int idt_idx = b*ih*iw*ic + sample*i*iw*ic + sample*j*ic + k*stride;
  output[out_idx] = input[out_idx] + identity[idt_idx];

  //output[index] = input[index] + identity[index];
}


void Shortcut::Forward() {

  size_t size = batch*oh*ow*oc;
  shortcut_forward_gpu_kernel<<<default_grid(size), BLOCK>>>(input, output, identity, size,
  iw, ih, ic, ow, oh, oc);
  check_error(cudaGetLastError());

}

__global__ void shortcut_backward_gpu_kernel(float *delta, float *delta_, float *cut, int size,
  int iw, int ih, int ic, int ow, int oh, int oc) {

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if( index >= size)
    return;

  int minw = (ow < iw) ? ow : iw;
  int minh = (oh < ih) ? oh : ih;
  int maxc = (oc > ic) ? oc : ic;

  int k = index%maxc;
  int j = index/maxc%minw;
  int i = index/maxc/minw%minh;
  int b = index/maxc/minw/minh;

  int sample = iw/ow;
  int stride = ic/oc;


  int out_idx = b*oh*ow*oc + i*ow*oc + j*oc + k;
  int idt_idx = b*ih*iw*ic + sample*i*iw*ic + sample*j*ic + k*stride;


  delta_[out_idx] = delta[out_idx];
  cut[idt_idx] = delta[out_idx];
}




void Shortcut::Backward(float *delta) {

  size_t size = batch*oh*ow*oc;
  shortcut_backward_gpu_kernel<<<default_grid(size), BLOCK>>>(delta, delta_, activation->cut, size,
  iw, ih, ic, ow, oh, oc);
  check_error(cudaGetLastError());

}

