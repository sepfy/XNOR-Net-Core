#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "network.h"
#include <byteswap.h>

#define IM_SIZE 28*28*1
#define CLASSES 10

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

#ifdef GPU
  float *X = malloc_gpu(N*M);
#else
  float *X = new float[N*M];
#endif
//  memset(X, 0, N*M*sizeof(float));

  uint8_t value;
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
      fread(&value, sizeof(uint8_t), 1, fp);
      //if(value > 0)
      //  X[i*M+j] = 1.0;
      //else
      //  X[i*M+j] = 0.0;
      //X[i*M+j] = ((float)value - 127.5)/127.5;
      X[i*M+j] = ((float)value)/255.0;
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
  
#ifdef GPU
  float *Y = malloc_gpu(N*10); 
#else
  float *Y = new float[N*10]; 
#endif
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

float* read_train_data(char *folder) {
  char filename[64] = {0};
  sprintf(filename, "%s/train-images-idx3-ubyte", folder);
  return read_images(filename);
}

float* read_train_label(char *folder) {
  char filename[64] = {0};
  sprintf(filename, "%s/train-labels-idx1-ubyte", folder);
  return read_labels(filename);
}


float* read_validate_data(char *folder) {
  char filename[64] = {0};
  sprintf(filename, "%s/t10k-images-idx3-ubyte", folder);
  return read_images(filename);
}

float* read_validate_label(char *folder) {
  char filename[64] = {0};
  sprintf(filename, "%s/t10k-labels-idx1-ubyte", folder);
  return read_labels(filename);
}

