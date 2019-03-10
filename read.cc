#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "read.h"

Tensor read_images(void) {

  FILE *fp;
  //FILE *label_fp;

  char image_name[] = "train-images-idx3-ubyte";
  //char label_name[] = "train-labels-idx3-ubyte";

  fp = fopen(image_name, "rb");
  uint32_t buff;

  int N, cols, rows;

  // Magic number
  fread(&buff, sizeof(uint32_t), 1, fp);
  printf("Magic number = %d\n", bswap_32(buff));

  // Number
  fread(&buff, sizeof(uint32_t), 1, fp);
  N = bswap_32(buff);
  N = 600;
  printf("Number of images = %d\n", N);
  // Rows
  fread(&buff, sizeof(uint32_t), 1, fp);
  rows = bswap_32(buff);
  printf("Number of rows = %d\n", rows);

  // Columns
  fread(&buff, sizeof(uint32_t), 1, fp);
  cols = bswap_32(buff);
  printf("Number of columns = %d\n", cols);
 
  Tensor X(N, rows*cols);
  uint8_t value;

  for(int i = 0; i < N; i++) {
    for(int j = 0; j < rows*cols; j++) {
      fread(&value, sizeof(uint8_t), 1, fp);
      if(value > 0)
        X.value[i][j] = 1.0;
      else
        X.value[i][j] = 0.0;
    }
  } 
  return X;
}
  
void show_image(Tensor X, int num) {
  for(int j = 0; j < 784; j++) {
    if(j%28==0)
      printf("\n");
    printf("%d", (int)X.value[num][j]);
  }
  printf("\n");
}


Tensor read_labels(void) {

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
  N = 600;
  printf("Number of images = %d\n", N);
 
  uint8_t value;
  Tensor Y(N, 10); 
  for(int i = 0; i < N; i++) {
    fread(&value, sizeof(uint8_t), 1, fp);
    for(int j = 0; j < 10; j++) {
      Y.value[i][j] = 0.0;
    }
    Y.value[i][(int)value] = 1.0;
  } 

  return Y;
}  

void show_label(Tensor Y, int row) {

  for(int j = 0; j < 10; j++) {
    printf("%d", (int)Y.value[row][j]);
  }
  printf("\n");
}

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
