#include <iostream>
#include "tensor.h"
#include "layers.h"
#include "gemm.h"
#include "blas.h"
using namespace std;

bool gemm_test() {
  cout << "Test gemm" << endl;
  int M = 3;
  int N = 2;
  int P = 3;
  float A[] = { 3, 4, 1, 2, 1, 3, 2, 2, 1}; //3x3
  float B[] = { 1, 2, 3, 2, 1, 2};  //3x2
  float D[] = {32, 32, 16, 24, 18, 20}; //3x2
  float alpha = 2;
  float *C = new float[N*P];
  memset(C, 0, N*P*sizeof(float));
  gemm(M, N, P, alpha, A, B, C);
/* 
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < P; j++) {
      cout << C[i*P+j] << " ";
    }
    cout << endl;
  }
*/
  cout << compare(M, N, C, D) << endl;
  
}

bool gemm_ta_test() {
  cout << "Test gemm_ta" << endl;

  int M = 2;
  int N = 2;
  int P = 3;
/*
 A = 3 4    A = 3 1 1
     1 2        4 2 3
     1 3

 B = 1 2
     3 2
     1 2

 D = 14 20
     26 36
*/
  float A[] = { 3, 4, 1, 2, 1, 3}; //3x2
  float B[] = { 1, 2, 3, 2, 1, 2}; //3x2
  float D[] = {14, 20, 26, 36}; // 2x2
  float alpha = 2;
  float *C = new float[N*P];
  memset(C, 0, N*P*sizeof(float));

  gemm_ta(M, N, P, alpha, A, B, C);
 
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      cout << C[i*N+j] << " ";
    }
    cout << endl;
  }
  cout << compare(M, N, C, D) << endl;
  
}

bool gemm_tb_test() {
  cout << "Test gemm_tb" << endl;

  int M = 3;
  int N = 3;
  int P = 2;
/*
 A = 3 4    
     1 2      
     1 3

 B = 1 2    B = 1 3 1
     3 2        2 2 2
     1 2

 D = 22 34 22
     10 14 10
     14 18 14
*/
  float A[] = { 3, 4, 1, 2, 1, 3}; //3x2
  float B[] = { 1, 2, 3, 2, 1, 2}; //3x2
  float D[] = {22, 34, 22, 10, 14, 10, 14, 18, 14}; // 3x3
  float alpha = 2;
  float *C = new float[N*P];
  memset(C, 0, N*P*sizeof(float));

  gemm_tb(M, N, P, alpha, A, B, C);
 
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      cout << C[i*N+j] << " ";
    }
    cout << endl;
  }
  cout << compare(M, N, C, D) << endl;
}  

int main() {

  /*
    A = 3 4 1
        2 1 3
        2 2 1

    B = 1 2
        3 2
        1 2 

    C = 32 32
        16 24
        18 20
  */

  gemm_test();
  gemm_ta_test();
  gemm_tb_test();
/*
  float b[] = { 1, 3, 2};
//  sgemm(N, M, P, alpha, A, B, C);
  bias_add(N, M, A, b);
//  for(int i = 0; i < N; i++) {
//    memcpy(C+i*P, b, sizeof(b));
//  }
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
      cout << A[i*M+j] << " ";
    }
    cout << endl;
  }
*/
  return 0;
}
