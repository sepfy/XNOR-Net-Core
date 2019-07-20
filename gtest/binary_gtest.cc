#include <iostream>
#include <random>
#include <sys/time.h>
#include <math.h>
#include <stdint.h>
#include "gemm.cc"
#include "string.h"
using namespace std;

#define BN 64

unsigned long long getms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec*1.0e+6 + tv.tv_usec;
}


int bitcount(uint64_t n){
  return  __builtin_popcountl(n);
}

class Bitset {

  public:
    uint64_t *bits;
    int bitnum;
    uint64_t complement;
    int offset;
    int N;

    void set(int idx, uint64_t value) {
      int bidx = idx/BN;
      int offset = idx%BN;
      bits[bidx] |= (value << offset);
    }

    Bitset() {}

    void init(int input_size, float *inputs) {
      bitnum = input_size;
      offset = input_size%BN;

      N = ceil((float)input_size/BN);
      bits = new uint64_t[N];
      memset(bits, 0, N*sizeof(uint64_t));
      complement = (UINT64_MAX << (BN - offset)) >> (BN - offset);

      for(int i = 0; i < input_size; i++) {
        //inputs[i] > 0 ? set(input_size-1-i, 1) : set(input_size-1-i, 0);
        inputs[i] > 0 ? set(i, 1) : set(i, 0);
      }
    }

};


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
  float alpha, float *A, float *B, float *C) {

  Bitset *bA = new Bitset[M];
  Bitset *bB = new Bitset[N];

  long long start = getms();  
  // Compress col
  for(int i = 0; i < M; i++) {
    bA[i].init(P, A+i*P);
  }
  cout << __LINE__ << ": " << getms() - start << endl;
  // Compress row
  // B = P*N
  float *BB = new float[N*P];
  start = getms();
  for(int i = 0; i < N; i++)
    for(int j = 0; j < P; j++)
      BB[i*P+j] = B[j*N+i];
  cout << __LINE__ << ": " << getms() - start << endl;
  start = getms();

  for(int i = 0; i < N; i++) {
    bB[i].init(P, BB+i*P);
  }

  cout << __LINE__ << ": " << getms() - start << endl;
  start = getms();

  #pragma omp parallel for 
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++)
      C[i*N+j] = popcount_xnor(bA+i, bB+j); 
  cout << __LINE__ << ": " << getms() - start << endl;
}

int main() {


  int M = 10;
  int N = 100;
  int P = 10240;

  float *A = new float[M*P];
  float *B = new float[P*N];
  float *C = new float[M*N];
  float *D = new float[M*N];

  default_random_engine generator = default_random_engine(time(NULL));
  normal_distribution<float> distribution(0, 0.5);

  for(int i = 0; i < M; i++) {
    for(int j = 0; j < P; j++) {
      A[i*P+j] = distribution(generator);
      A[i*P+j] > 0 ? A[i*P+j] = 1.0 : A[i*P+j] = -1.0;
    }
  }

  for(int i = 0; i < P; i++) {
    for(int j = 0; j < N; j++) {
      B[i*N+j] = distribution(generator);
      B[i*N+j] > 0 ? B[i*N+j] = 1.0 : B[i*N+j] = -1.0;
    }
  }

  long long int start = getms();  
  gemm(M, N, P, 1.0, A, B, C);
  cout << getms() - start << endl;


  start = getms();  
  bin_gemm(M, N, P, 1.0, A, B, D);
  cout << getms() - start << endl;

  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      if(C[i*N+j] != D[i*N+j])
        cout << i << "," << j << endl;
    }
  }
  return 0;
}
