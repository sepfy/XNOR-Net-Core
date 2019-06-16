#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "blas.h"

using namespace std;


void add(int N, int M, float *A, float *B, float *C) {
  int idx;
  for(int i = 0; i < N; i++)
    for(int j = 0; j < M; j++) {
      idx = i*M + j;
      C[idx] = A[idx] + B[idx];
    }
}

void bias_add(int batch, int M, float *A, float *bias) {
  
  for(int i = 0; i < batch; i++)
    add(1, M, A+i*M, bias, A+i*M);
}


/*
void bias_add(int N, int M, float *A, float *bias) {

    size_t length = M*sizeof(float);
    for(int i = 0; i < N; i++)
      memcpy(A+i*M, bias, length);
}
*/

void col_sum(int N, int M, float *A, float *B) {
  memset(B, 0, N*sizeof(float));
  for(int i = 0; i < N; i++)
    for(int j = 0; j < M; j++)
      B[i] += A[i*M+j];
}


void row_sum(int N, int M, float *A, float *B) {
  memset(B, 0, M*sizeof(float));
  for(int j = 0; j < M; j++)
    for(int i = 0; i < N; i++)
      B[j] += A[i*M+j];
}


bool compare(int N, int M, float *A, float *B) {

  int len = N*M;
  for(int i = 0; i < len; i++)
    if(A[i] != B[i])
      return false;
  return true;
}

void mat_minus(int N, int M, float *mat1, float *mat2, float* mat_out) {

  int length = N*M;
  for(int i = 0; i < length; i++)
     mat_out[i] = mat1[i] - mat2[i]; 
}

void mat_scalar(int N, int M, float *mat1, float scalar, float* mat_out) {

  int length = N*M;
  for(int i = 0; i < length; i++)
     mat_out[i] = mat1[i]*scalar; 
}

float cross_entropy(int batch, int N, float *output, float *target) {

  float tmp = 0;
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      tmp -= output[i*N+j]*log(target[i*N+j] + 1.0e-6)
        + (1.0 - output[i*N+j])*log(1.0 - target[i*N+j] + 1.0e-6);
    }
  }
  tmp = tmp/(float)batch;
  return tmp;
}

float L1_norm(int N, int M, float *A) {

  float norm = 0;
  for(int i = 0; i < N; i++) 
    for(int j = 0; j < M; j++) 
      norm += fabs(A[i*M+j]);

  return norm;
}

float L2_norm(int N, int M, float *A) {

  float norm = 0;
  for(int i = 0; i < N; i++) 
    for(int j = 0; j < M; j++) 
      norm += pow(A[i*M+j], 2.0);

  return norm;
}

float Linf_norm(int N, int M, float *A) {

  float max = -1.0e5;
  for(int i = 0; i < N; i++)
    for(int j = 0; j < M; j++)
      if(A[i*M+j] > max)
        max = A[i*M+j];
  return max;
}


void transpose(int N, int M, float *A, float *B) {

  for(int j = 0; j < M; j++)
    for(int i = 0; i < N; i++)
      B[j*N+i] = A[i*M+j];   
}
