#include <iostream>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "utils.h"
#include "binary.h"
using namespace std;

void Bitset::init(int input_size) {
  BN = sizeof(uint64_t)*8;
  bitnum = input_size;
  offset = input_size%BN;

  N = ceil((float)input_size/BN);
  bits = new uint64_t[N];
  memset(bits, 0, sizeof(uint64_t)*N);
  complement = (UINT64_MAX << (BN - offset)) >> (BN - offset);
}

void Bitset::set(float *inputs) {
  for(int i = 0; i < bitnum; i++) {
    int bidx = i/BN;
    int offset = i%BN;
    if(inputs[i] > 0) {
      uint64_t value = 1;
      bits[bidx] |= ((uint64_t)1 << offset);
    }
    else { 
      uint64_t value = 1;
      bits[bidx] |= ((uint64_t)0 << offset);
    }
  }
}


int bitcount(uint64_t n){
  return  __builtin_popcountl(n);
}

float popcount_xnor(Bitset *b1, Bitset *b2) {

  int c = 0;
  for(int i = 0; i < b1->N - 1; i++) {
    uint64_t tmp = ~(b1->bits[i]^b2->bits[i]);
    c += bitcount(tmp);
  } 
  uint64_t tmp = ~(b1->bits[b1->N-1]^b2->bits[b1->N-1]);
  tmp &= b1->complement;
  c += bitcount(tmp);
  
  return (float)(2*c - b1->bitnum);
}


void bin_gemm(int M, int N, int P,
  float alpha, Bitset *bA, Bitset *bB, float *C) {

ms_t start = getms();
  #pragma omp parallel for 
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++)
      C[i*N+j] = popcount_xnor(bA+i, bB+j); 
cout << "popcount time = " << getms() -start << endl;
}


void bin_gemm(int M, int N, int P,
  float alpha, float *A, float *B, float *C) {


  Bitset *bA = new Bitset[M];
  Bitset *bB = new Bitset[N];

  for(int i = 0; i < M; i++) {
    bA[i].init(P);
    bA[i].set(A+i*P);
  }

  float *BB = new float[N*P];
  for(int i = 0; i < N; i++)
    for(int j = 0; j < P; j++)
      BB[i*P+j] = B[j*N+i];

  for(int i = 0; i < N; i++) {
    bB[i].init(P);
    bB[i].set(BB+i*P);
  }

  bin_gemm(M, N, P, alpha, bA, bB, C);

}

