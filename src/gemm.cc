#include <iostream>
#include <omp.h>
#include "gemm.h"
#include <string.h>
using namespace std;
/*
  C = MxN, A = MxP, B = PxN
*/
void gemm(int M, int N, int P,
  float alpha, float *A, float *B, float *C) {

  memset(C, 0, M*N*sizeof(float));
  #pragma omp parallel for
  for(int i = 0; i < M; i++)
    for(int k = 0; k < P; k++) {
      float A_PART = alpha*A[i*P+k];
      for(int j = 0; j < N; j++) 
        C[i*N+j] += A_PART*B[k*N+j];
    }
}

/*
  C = MxN, A = PxM, B = PxN
*/
void gemm_ta(int M, int N, int P,
  float alpha, float *A, float *B, float *C) {

#if 1
  // I am not sure why this procedure is faster than the following procedure.
  // If anyone konws, please tell me! Thanks... 
  #pragma omp parallel for
  for(int i = 0; i < M; i++)
    for(int k = 0; k < P; k++)
      for(int j = 0; j < N; j++) {
        C[i*N+j] += alpha*A[k*M+i]*B[k*N+j];
    }
#endif
#if 0
  #pragma omp parallel for
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++) {
      C[i*N+j] = 0;
      for(int k = 0; k < P; k++)
        C[i*N+j] += A[k*M+i]*B[k*N+j];
      C[i*N+j] *= alpha;
    }
#endif

}

/*
  C = MxN, A = MxP, B = NxP
*/
void gemm_tb(int M, int N, int P,
  float alpha, float *A, float *B, float *C) {

  #pragma omp parallel for
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++) {
      C[i*N+j] = 0;
      for(int k = 0; k < P; k++)
        C[i*N+j] += A[i*P+k]*B[j*P+k];
      C[i*N+j] *= alpha;
    }
}
