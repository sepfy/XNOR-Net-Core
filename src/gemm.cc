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

#if 1
  // I am not sure why this procedure is faster than the following procedure.
  // If anyone konws, please tell me! Thanks... 
#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
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


void gemm_beta(int M, int N, int P,
  float alpha, float *A, float *B, float *C) {

//  memset(C, 0, M*N*sizeof(float));
#ifdef USE_OPENMP
  #pragma omp parallel for
#endif

  for(int i = 0; i < M; i+=4) {

    for(int j = 0; j < N; j++) C[i*N+j] = 0.0; 
    for(int j = 0; j < N; j++) C[(i+1)*N+j] = 0.0; 
    for(int j = 0; j < N; j++) C[(i+2)*N+j] = 0.0; 
    for(int j = 0; j < N; j++) C[(i+3)*N+j] = 0.0; 

    for(int k = 0; k < P; k+=4) {
      int idx = i*P+k;
      register float A_PART = A[idx];
      register float B_PART = A[idx+1];
      register float C_PART = A[idx+2];
      register float D_PART = A[idx+3];

      register float E_PART = A[(i+1)*P+k];
      register float F_PART = A[(i+1)*P+k+1];
      register float G_PART = A[(i+1)*P+k+2];
      register float H_PART = A[(i+1)*P+k+3];

      register float I_PART = A[(i+2)*P+k];
      register float J_PART = A[(i+2)*P+k+1];
      register float K_PART = A[(i+2)*P+k+2];
      register float L_PART = A[(i+2)*P+k+3];

      register float M_PART = A[(i+3)*P+k];
      register float N_PART = A[(i+3)*P+k+1];
      register float O_PART = A[(i+3)*P+k+2];
      register float P_PART = A[(i+3)*P+k+3];


   for(int j = 0; j < N; j++) 
  C[i*N+j] += (A_PART*B[k*N+j] + B_PART*B[(k+1)*N+j] + C_PART*B[(k+2)*N+j] + D_PART*B[(k+3)*N+j]);
      
      for(int j = 0; j < N; j++) 
 C[(i+1)*N+j] += (E_PART*B[k*N+j] + F_PART*B[(k+1)*N+j] + G_PART*B[(k+2)*N+j] + H_PART*B[(k+3)*N+j]);

      for(int j = 0; j < N; j++) 
 C[(i+2)*N+j] += (I_PART*B[k*N+j] + J_PART*B[(k+1)*N+j] + K_PART*B[(k+2)*N+j] + L_PART*B[(k+3)*N+j]);
      for(int j = 0; j < N; j++) 
 C[(i+3)*N+j] += (M_PART*B[k*N+j] + N_PART*B[(k+1)*N+j] + O_PART*B[(k+2)*N+j] + P_PART*B[(k+3)*N+j]);



      }

    }


/*
  for(int j = 0; j < N; j+=4) {
    for(int i = 0; i < M; i+=4) {


      register float c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34, c41, c42, c43, c44;
      c11 = 0.0;
      c12 = 0.0;
      c13 = 0.0;
      c14 = 0.0;
      c21 = 0.0;
      c22 = 0.0;
      c23 = 0.0;
      c24 = 0.0;
      c31 = 0.0;
      c32 = 0.0;
      c33 = 0.0;
      c34 = 0.0;
      c41 = 0.0;
      c42 = 0.0;
      c43 = 0.0;
      c44 = 0.0;

      register float a1, a2, a3, a4;

      register float b1, b2, b3, b4;

      for(int k = 0; k < P; k++) {
         a1 = A[i*P+k];
         a2 = A[(i+1)*P+k];
         a3 = A[(i+2)*P+k];
         a4 = A[(i+3)*P+k];

         b1 = B[k*N+j];
         b2 = B[k*N+j+1];
         b3 = B[k*N+j+2];
         b4 = B[k*N+j+3];

         c11 += a1*b1;
         c21 += a2*b1;

         c12 += a1*b2;
         c22 += a2*b2;

         c13 += a1*b3;
         c23 += a2*b3;

         c14 += a1*b4;
         c24 += a2*b4;

	 c31 += a3*b1;
         c41 += a4*b1;

	 c32 += a3*b2;
         c42 += a4*b2;

	 c33 += a3*b3;
         c43 += a4*b3;

	 c34 += a3*b4;
         c44 += a4*b4;

      }
        C[i*N+j] = c11;
        C[(i+1)*N+j] = c21;
        C[(i+2)*N+j] = c31;
        C[(i+3)*N+j] = c41;

      
        C[i*N+j+1] = c12;
        C[(i+1)*N+j+1] = c22;
        C[(i+2)*N+j+1] = c23;
        C[(i+3)*N+j+1] = c24;

        C[i*N+j+2] = c31;
        C[(i+1)*N+j+2] = c32;
        C[(i+2)*N+j+2] = c33;
        C[(i+3)*N+j+2] = c34;

        C[i*N+j+3] = c41;
        C[(i+1)*N+j+3] = c42;
        C[(i+2)*N+j+3] = c43;
        C[(i+3)*N+j+3] = c44; 
    }
  }
  */
}


