#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "blas.h"

using namespace std;

int* argmax(int batch, int N, float *A) {

  int idx = 0;
  float max = 0;
  int *output = new int[batch];

  for(int i = 0; i < batch; i++) {
    max = 0;
    for(int j = 0; j < N; j++)
      if(A[i*N+j] > max) {
        max = A[i*N+j];
        idx = j;
      }
      output[i] = idx;
  }

  return output;
}

float accuracy(int batch, int N, float *A, float *B) {
 
  int *argA = argmax(batch, N, A);
  int *argB = argmax(batch, N, B);

  float sum = 0;
  for(int i = 0; i < batch; i++)
    if(argA[i] != argB[i])
      sum += 1.0;
  sum = sum/(float)batch;
  delete []argA;
  delete []argB;
  return sum;
}

