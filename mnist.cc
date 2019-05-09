#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "network.h"
#include "loss.h"
#include <byteswap.h>

#if 0
#include <libkern/OSByteOrder.h>
#define bswap_16(x) OSSwapInt16(x)
#define bswap_32(x) OSSwapInt32(x)
#define bswap_64(x) OSSwapInt64(x)
#endif

int batch = 600;
float* read_images(char *filename) {

  FILE *fp;

  fp = fopen(filename, "rb");
  if(!fp) {
    cout << "Cannot open file " << filename << endl;
    exit(1);
  }

  int N, M, cols, rows;
  uint32_t buff;

  // Magic number
  fread(&buff, sizeof(uint32_t), 1, fp);
  printf("Magic number = %d\n", bswap_32(buff));

  // Number
  fread(&buff, sizeof(uint32_t), 1, fp);
  N = bswap_32(buff);
  N = batch;
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


float* read_labels(char *filename) {

  FILE *fp;
  //FILE *label_fp;

  fp = fopen(filename, "rb");
  if(!fp) {
    cout << "Cannot open file " << filename << endl;
    exit(1);
  }

  uint32_t buff;
  int N, cols, rows;

  // Magic number
  fread(&buff, sizeof(uint32_t), 1, fp);
  printf("Magic number = %d\n", bswap_32(buff));

  // Number
  fread(&buff, sizeof(uint32_t), 1, fp);
  N = bswap_32(buff);
  N = batch;
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



float* read_train_data() {
  char filename[] = "train-images-idx3-ubyte";
  return read_images(filename);
}

float* read_train_label() {
  char filename[] = "train-labels-idx1-ubyte";
  return read_labels(filename);
}


float* read_validate_data() {
  char filename[] = "t10k-images-idx3-ubyte";
  return read_images(filename);
}

float* read_validate_label() {
  char filename[] = "t10k-labels-idx1-ubyte";
  return read_labels(filename);
}

/*
float validate() {


}
*/


int main(void) {
  float *X, *Y;
  
  X = read_train_data();
  Y = read_train_label();
  //show_image(X, 2);
  //show_label(Y, 2);


  Convolution conv1(batch, 28, 28, 1, 5, 5, 3, 1, 2, X);
  // 28 + 2*2 - 5)/1 + 1 = 28
  Relu relu1(batch, 28*28*3, conv1.output);
  Pooling pool1(batch, 28, 28, 3, 2, 2, 3, 2, 0, relu1.output); 

  Convolution conv2(batch, 14, 14, 3, 3, 3, 3, 1, 1, pool1.output);
  // (14 + 2*1 - 3)/1 + 1 = 12
  Relu relu2(batch, 14*14*3, conv2.output);
  Pooling pool2(batch, 14, 14, 3, 2, 2, 3, 2, 0, relu2.output); 




  Connected conn1(batch, 7*7*3, 10, pool2.output);
  //Connected conn1(batch, 784, 10, X);
  SoftmaxWithCrossEntropy softmax(batch, 10, Y, conn1.output);

  Network network;
  network.add(&conv1);
  network.add(&relu1);
  network.add(&pool1);
  network.add(&conv2);
  network.add(&relu2);
  network.add(&pool2);
  network.add(&conn1);
  network.add(&softmax);
  int max_iter = 500;
  for(int iter = 0; iter < max_iter; iter++) {
    network.inference();
    cout << "iter = " << iter << ", accuracy = "
         << accuracy(batch, 10, softmax.output, Y) << endl;
    network.train(Y);
  }

  
  X = read_validate_data();
  Y = read_validate_label();
  conv1.input = X;
  network.inference();
  cout << "Validate set accuracy = "
       << accuracy(batch, 10, softmax.output, Y) << endl;
  


  return 0;
}


