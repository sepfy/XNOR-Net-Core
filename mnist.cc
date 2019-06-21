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

#if 0
  Connected conn1(28*28, 256);
  Connected conn2(256, 256);
  Connected conn3(256, 10);
  SoftmaxWithCrossEntropy softmax(10, Y);
  Network network;
  network.add(&conn1);
  network.add(&conn2);
  network.add(&conn3);
  network.add(&softmax);
#endif
#if 1
  Convolution conv1(28, 28, 1, 5, 5, 3, 1, true);
  Relu relu1(28*28*3);
  Pooling pool1(28, 28, 3, 2, 2, 3, 2, false); 
  Convolution conv2(14, 14, 3, 3, 3, 3, 1, true);
  Relu relu2(14*14*3);
  Pooling pool2(14, 14, 3, 2, 2, 3, 2, false); 
  Connected conn1(7*7*3, 10);
  SoftmaxWithCrossEntropy softmax(10);

  Network network;
  network.add(&conv1);
  network.add(&relu1);
  network.add(&pool1);
  network.add(&conv2);
  network.add(&relu2);
  network.add(&pool2);
  network.add(&conn1);
  network.add(&softmax);
#endif

  int max_iter = 100000;
  float total_err = 0;


  int batch = 100;
  int epoch = 10;

  network.initial(batch, .1);

 
  for(int iter = 0; iter < max_iter; iter++) {

    //int step = (iter*batch)%60000;
    int step = 0;
    float *batch_xs = X + step*784;
    float *batch_ys = Y + step*10;

    ms_t start = getms();
    float *output = network.inference(batch_xs);
    network.train(batch_ys);// + step);
 
    total_err = accuracy(batch, 10, output, batch_ys);// + step);

    //if(step == 0) {
      cout << "iter = " << iter << ", time = " << (getms() - start) << "ms, error = "
       << total_err << endl;
      if(total_err < 0.001)
        break;
//      total_err = 0.0;
//    }
  }

  
  X = read_validate_data();
  Y = read_validate_label();
  float *output = network.inference(X);
  total_err = accuracy(batch, 10, output, Y);
  cout << "Validate set error = " << total_err << endl;
  


  return 0;
}


