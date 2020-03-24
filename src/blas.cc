#include <iostream>
#include <stdlib.h>
#include <string.h>
#ifdef GPU
#include "gpu.h"
#endif
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


#ifdef GPU

float cross_entropy(int batch, int N, float *output_gpu, float *target_gpu) {
  float *output = new float[batch*N];
  float *target = new float[batch*N];

  gpu_pull_array(output_gpu, output, batch*N);
  gpu_pull_array(target_gpu, target, batch*N);

  float tmp = 0;
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      if(target[i*N+j] == 1)
	      tmp -= log(output[i*N+j] + 1.0e-6);
      //tmp -= output[i*N+j]*log(target[i*N+j] + 1.0e-6)
      //  + (1.0 - output[i*N+j])*log(1.0 - target[i*N+j] + 1.0e-6);
    }
  }
  tmp = tmp/(float)batch;

  delete []output;
  delete []target;

  return tmp;
}

#else

float cross_entropy(int batch, int N, float *output, float *target) {
  float tmp = 0;
  for(int i = 0; i < batch; i++) {
    for(int j = 0; j < N; j++) {
      if(target[i*N+j] == 1)
	      tmp -= log(output[i*N+j] + 1.0e-6);
      //tmp -= output[i*N+j]*log(target[i*N+j] + 1.0e-6)
      //  + (1.0 - output[i*N+j])*log(1.0 - target[i*N+j] + 1.0e-6);
    }
  }
  tmp = tmp/(float)batch;

  return tmp;
}
#endif


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


void scalar(int N, float s, float *A, float *B) {

  for(int i = 0; i < N; i++)
    B[i] = A[i]*s;
}


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

#ifdef GPU

float axpy_gpu(int size, float alpha, float *A, float *B) {
  cublasSaxpy(gpu_handle(), size, &alpha, A, 1, B, 1);
  check_error(cudaGetLastError());
}

float accuracy(int batch, int N, float *A_gpu, float *B_gpu) {

  float *A = new float[batch*N];
  float *B = new float[batch*N];

  gpu_pull_array(A_gpu, A, batch*N);
  gpu_pull_array(B_gpu, B, batch*N);
  int *argA = argmax(batch, N, A);
  int *argB = argmax(batch, N, B);

  float sum = 0;
  for(int i = 0; i < batch; i++)
    if(argA[i] == argB[i])
      sum += 1.0;
  sum = sum/(float)batch;
  delete []argA;
  delete []argB;
  delete []A;
  delete []B;
  return sum;
}
#else
float accuracy(int batch, int N, float *A, float *B) {

  int *argA = argmax(batch, N, A);
  int *argB = argmax(batch, N, B);

  float sum = 0;
  for(int i = 0; i < batch; i++)
    if(argA[i] == argB[i])
      sum += 1.0;
  sum = sum/(float)batch;
  delete []argA;
  delete []argB;
  return sum;
}
#endif
