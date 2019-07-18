#include <iostream>
#include <random>
#include <sys/time.h>
#include <math.h>
#include <stdint.h>
using namespace std;

#define BN 32

int bitcount(uint32_t n){
  return  __builtin_popcountl(n);
}

class Bitset {

  public:
    uint32_t *bits;
    int bitnum;
    uint32_t complement;
    int offset;
    int N;

    void set(int idx, uint32_t value) {
      int bidx = idx/BN;
      int offset = idx%BN;
      bits[bidx] |= (value << offset);
    }

    Bitset() {}

    void init(int input_size, float *inputs) {
      bitnum = input_size;
      offset = input_size%BN;

      N = ceil((float)input_size/BN);
      bits = new uint32_t[N];
      memset(bits, 0, sizeof(uint32_t)*N);
      complement = (UINT32_MAX << (BN - offset)) >> (BN - offset);

      for(int i = 0; i < input_size; i++) {
        //inputs[i] > 0 ? set(input_size-1-i, 1) : set(input_size-1-i, 0);
        inputs[i] > 0 ? set(i, 1) : set(i, 0);
      }
    }

};


float popcount_xnor(Bitset *b1, Bitset *b2) {

  int c = 0;
  for(int i = 0; i < b1->N - 1; i++) {
    uint32_t tmp = ~(b1->bits[i]^b2->bits[i]);
    c += bitcount(tmp);
  } 
  uint32_t tmp = ~(b1->bits[b1->N-1]^b2->bits[b1->N-1]);
  tmp &= b1->complement;
  c += bitcount(tmp);
  
  return (float)(2*c - b1->bitnum);
}

void bin_gemm(int M, int N, int P,
  float alpha, float *A, float *B, float *C) {

  Bitset *bA = new Bitset[M];
  Bitset *bB = new Bitset[N];

  // Compress col
  for(int i = 0; i < M; i++) {
    bA[i].init(P, A+i*P);
  }
  // Compress row
  // B = P*N
  float *BB = new float[N*P];
  for(int i = 0; i < N; i++)
    for(int j = 0; j < P; j++)
      BB[i*P+j] = B[j*N+i];

  for(int i = 0; i < N; i++) {
    bB[i].init(P, BB+i*P);
  }


  #pragma omp parallel for 
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++)
      C[i*N+j] = popcount_xnor(bA+i, bB+j); 
}

