#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <byteswap.h>
//#include <libkern/OSByteOrder.h>
//#define bswap_16(x) OSSwapInt16(x)
//#define bswap_32(x) OSSwapInt32(x)
//#define bswap_64(x) OSSwapInt64(x)
#include "tensor.h"

Tensor read_images(void);
  
void show_image(Tensor X, int num);

Tensor read_labels(void);

void show_label(Tensor Y, int row) ;


/*
int main(void) {
  double **X, **Y;
  
  X = read_images();
  Y = read_labels();
  show_image(X, 60);
  show_label(Y, 60);
  return 0;
}

*/
