#include "layers.h"

#ifdef GPU
#include "gpu.h"
#endif

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
  cudaDeviceSynchronize();
}




