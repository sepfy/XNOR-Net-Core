#ifndef GEMM_H_
#define GEMM_H_

enum TRS {
  TRS_N,
  TRS_T
};

void gemm_cpu(TRS TRS_A, TRS TRS_B,
           int M, int N, int P,
  float alpha, float *A, float *B, float *C);

void gemm(int M, int N, int P,
  float alpha, float *A, float *B, float *C);

void gemm_ta(int M, int N, int P,
  float alpha, float *A, float *B, float *C); 

void gemm_tb(int M, int N, int P,
  float alpha, float *A, float *B, float *C);

void gemm_gpu(TRS TRS_A, TRS TRS_B,
              int M, int N, int P,
  float alpha, float *A, float *B, float *C);

#endif //  GEMM_H_
