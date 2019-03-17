#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "tensor.h"
#include "layers.h"
#include "loss.h"
#include <byteswap.h>
//#include <libkern/OSByteOrder.h>
//#define bswap_16(x) OSSwapInt16(x)
//#define bswap_32(x) OSSwapInt32(x)
//#define bswap_64(x) OSSwapInt64(x)


float* read_images() {

  FILE *fp;
  char image_name[] = "train-images-idx3-ubyte";

  fp = fopen(image_name, "rb");

  int N, M, cols, rows;
  uint32_t buff;

  // Magic number
  fread(&buff, sizeof(uint32_t), 1, fp);
  printf("Magic number = %d\n", bswap_32(buff));

  // Number
  fread(&buff, sizeof(uint32_t), 1, fp);
  N = bswap_32(buff);
  //N = 600;
  printf("Number of images = %d\n", N);
  // Rows
  fread(&buff, sizeof(uint32_t), 1, fp);
  rows = bswap_32(buff);
  printf("Number of rows = %d\n", rows);

  // Columns
  fread(&buff, sizeof(uint32_t), 1, fp);
  cols = bswap_32(buff);
  printf("Number of columns = %d\n", cols);
 
  M = rows*cols;

  float *X = new float[N*M];
//  memset(X, 0, N*M*sizeof(float));

  uint8_t value;
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
      fread(&value, sizeof(uint8_t), 1, fp);
      if(value > 0)
        X[i*M+j] = 1.0;
      else
        X[i*M+j] = 0.0;
    }
  }
  return X;
}
 
void show_image(float *X, int num) {

  for(int j = 0; j < 784; j++) {
    if(j%28==0)
      printf("\n");
    printf("%d", (int)X[num*784+j]);
  }
  printf("\n");
}


float* read_labels() {

  FILE *fp;
  //FILE *label_fp;

  char label_name[] = "train-labels-idx1-ubyte";

  fp = fopen(label_name, "rb");
  uint32_t buff;

  int N, cols, rows;

  // Magic number
  fread(&buff, sizeof(uint32_t), 1, fp);
  printf("Magic number = %d\n", bswap_32(buff));

  // Number
  fread(&buff, sizeof(uint32_t), 1, fp);
  N = bswap_32(buff);
//  N = 600;
  printf("Number of images = %d\n", N);
  
  float *Y = new float[N*10]; 
  uint8_t value;

  for(int i = 0; i < N; i++) {
    fread(&value, sizeof(uint8_t), 1, fp);
    for(int j = 0; j < 10; j++) {
      Y[i*10+j] = 0.0;
    }
    Y[i*10+(int)value] = 1.0;
  } 

  return Y;
}  


void show_label(float *Y, int num) {

  for(int j = 0; j < 10; j++) {
    printf("%d", (int)Y[num*10+j]);
  }
  printf("\n");
}


int main(void) {
  float *X, *Y;
  
  X = read_images();
  Y = read_labels();
  //show_image(X, 2);
  //show_label(Y, 2);
  int batch = 60000;

  Connected conn1(batch, 784, 10, X);
  Sigmoid sigmoid1(batch, 10, conn1.output);
  Connected conn2(batch, 10, 10, sigmoid1.output);
  SoftmaxWithCrossEntropy softmax(batch, 10, Y, conn2.output);

  int max_iter = 1000;
  for(int iter = 0; iter < max_iter; iter++) {

    conn1.forward();
    sigmoid1.forward();
    conn2.forward();
    softmax.forward();

    cout << "iter = " << iter << ", accuracy = "
         << accuracy(batch, 10, softmax.output, Y) << endl;

    softmax.backward(Y);
    conn2.backward(softmax.m_delta);
    sigmoid1.backward(conn2.m_delta);
    conn1.backward(sigmoid1.m_delta);



    conn2.update();
    conn1.update();
  }
  return 0;
}


