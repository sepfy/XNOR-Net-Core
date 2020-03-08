#include "layers.h"

#if 1
__global__ void bias_add_kernel(float *output, float *bias,
                         int out_h, int out_w, int FC, int size) {


    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index > size) return;

    int b = index/(out_h*out_w*FC);
    int i = index/(out_w*FC)%out_h ;
    int j = index/FC%out_w;
    int k = index%FC;

    output[b*out_w*out_h*FC + i*out_w*FC + j*FC +k] += bias[k];

}

void Convolution::bias_add_gpu() {

  size_t size = out_w*out_h*batch*FC;
  bias_add_kernel<<<default_grid(size), BLOCK>>>(output, bias, out_w, out_h, FC, size);
  check_error(cudaGetLastError());
}
#endif

#if 0
__global__ void bias_add_kernel(float *output, float *bias,
                         int batch, int im_size, int channel) {

    int i = threadIdx.x;
    int b = blockIdx.x;
    for(int j = 0; j < channel; j++)
      output[b*im_size*channel+i*channel+j] += bias[j];

}

void Convolution::bias_add_gpu() {

  int size = out_w*out_h;
  bias_add_kernel<<<batch, size>>>(output, bias, batch, size, FC);
  check_error(cudaGetLastError());
}

#endif


