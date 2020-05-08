#include <iostream>
#include <string.h>

#include "gemm.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif

/*
  C = MxN, A = MxP, B = PxN
*/

void gemm_cpu(TRS TRS_A, TRS TRS_B,
           int M, int N, int P,
  float alpha, float *A, float *B, float *C) {

  if(!TRS_A && !TRS_B)
    gemm(M, N, P, alpha, A, B, C);
  else if(TRS_A && !TRS_B)
    gemm_ta(M, N, P, alpha, A, B, C);
  else if(!TRS_A && TRS_B)
    gemm_tb(M, N, P, alpha, A, B, C);

}

void gemm(int M, int N, int P,
  float alpha, float *A, float *B, float *C) {

  memset(C, 0, M*N*sizeof(float));
#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
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

  memset(C, 0, M*N*sizeof(float));
#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for(int i = 0; i < M; i++)
    for(int k = 0; k < P; k++)
      for(int j = 0; j < N; j++) {
        C[i*N+j] += alpha*A[k*M+i]*B[k*N+j];
    }

}

/*
  C = MxN, A = MxP, B = NxP
*/
void gemm_tb(int M, int N, int P,
  float alpha, float *A, float *B, float *C) {

#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++) {
      C[i*N+j] = 0;
      for(int k = 0; k < P; k++)
        C[i*N+j] += A[i*P+k]*B[j*P+k];
      C[i*N+j] *= alpha;
    }
}

#ifdef GPU
#include "gpu.h"

void gemm_gpu(TRS TRS_A, TRS TRS_B,
              int M, int N, int P,
  float alpha, float *A, float *B, float *C) {


  float beta = 0.0;

  if(!TRS_A && !TRS_B)
    cublasSgemm(gpu_handle(),
              CUBLAS_OP_N, CUBLAS_OP_N,
	      N, M, P, &alpha, B, N, A, P, &beta, C, N);
  else if(TRS_A && !TRS_B)
    cublasSgemm(gpu_handle(),
              CUBLAS_OP_N, CUBLAS_OP_T,
	      N, M, P, &alpha, B, N, A, M, &beta, C, N);

  else if(!TRS_A && TRS_B)
    cublasSgemm(gpu_handle(),
              CUBLAS_OP_T, CUBLAS_OP_N,
	      N, M, P, &alpha, B, P, A, P, &beta, C, N);

  check_error(cudaGetLastError());
}

#endif
