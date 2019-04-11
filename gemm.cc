#include <iostream>
#include <omp.h>
#include "gemm.h"

using namespace std;
/*
  C = MxN, A = MxP, B = PxN
*/
void gemm(int M, int N, int P,
  float alpha, float *A, float *B, float *C) {

  #pragma omp parallel for
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++) {
      C[i*N+j] = 0.0;
      for(int k = 0; k < P; k++)
        C[i*N+j] += A[i*P+k]*B[k*N+j];
      C[i*N+j] *= alpha;
    }
}

/*
  C = MxN, A = PxM, B = PxN
*/
void gemm_ta(int M, int N, int P,
  float alpha, float *A, float *B, float *C) {
  
  #pragma omp parallel for
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++) {
      C[i*N+j] = 0;
      for(int k = 0; k < P; k++)
        C[i*N+j] += A[k*M+i]*B[k*N+j];
      C[i*N+j] *= alpha;
    }
}

/*
  C = MxN, A = MxP, B = NxP
*/
void gemm_tb(int M, int N, int P,
  float alpha, float *A, float *B, float *C) {

  #pragma omp parallel for
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++) {
      for(int k = 0; k < P; k++)
        C[i*N+j] += A[i*P+k]*B[j*P+k];
      C[i*N+j] *= alpha;
    }
}
