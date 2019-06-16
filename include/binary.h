#include <iostream>
#include <random>
#include <sys/time.h>
#include <math.h>

using namespace std;

float binarize(int size, float *input) {

  float l1 = 0.0;
  for(int i = 0; i < size; i++) {
    l1 += fabs(input[i]);
    input[i] >= 0 ? input[i] = 1 : input[i] = -1;
  }
  return l1/(float)size;
}



int bitcount(uint32_t n){
  return  __builtin_popcountl(n);
}

class Bitset {

  public:
    uint32_t *bits;
    int size;
    int bitnum;
    uint32_t complement;
    int row, col;
    int offset;
    float l1 = 0;
    float alpha = 0;
    Bitset(int _size) {
      bitnum = _size;
      size = ceil((float)_size/(8*sizeof(uint32_t)));
      bits = new uint32_t[size];
      offset = _size%(8*sizeof(uint32_t));
      complement = (UINT32_MAX << (32 - offset)) >> (32-offset);
    }

     
    // if tr is true, the compress with column
    // PxN -> bitset: N x \hat{P}
    // else
    // MxP -> bitset: M x \hat{P}
    Bitset(int N, int M, float *input, bool tr) {

      if(tr) {
        bitnum = N;
        offset = bitnum%(8*sizeof(uint32_t));
        size = ceil((float)bitnum/(8*sizeof(uint32_t)));
        bits = new uint32_t[M*size];
        for(int j = 0; j < M; j++) {
          for(int i = 0; i < N; i++) {
            int idx = j*size*(8*sizeof(uint32_t)) + N-1-i;
            input[i*M+j] >= 0 ? set(idx, 1) : set(idx, 0);
            l1 += fabs(input[i*M+j]);
          }
        }
        alpha = l1/(float)(N*M);
      }
      else {
        bitnum = M;
        offset = bitnum%(8*sizeof(uint32_t));
        size = ceil((float)bitnum/(8*sizeof(uint32_t)));
        bits = new uint32_t[N*size];
        for(int i = 0; i < N; i++) {
          for(int j = 0; j < M; j++) {
            int idx = i*size*(8*sizeof(uint32_t)) + M-1-j;
            input[i*M+j] >= 0 ? set(idx, 1) : set(idx, 0);
          }
        }
      }
      cout << alpha << endl;  
    }


    void set(int idx, uint32_t value) {
      int bidx = idx/(8*sizeof(uint32_t));
      int offset = idx%(8*sizeof(uint32_t));
      //cout << (value << offset)<<endl;
      bits[bidx] |= (value << offset);
    }

    void sign_to_bin(int size, float *input) {
      for(int i = 0; i < size; i++) {
        input[i] >= 0 ? set(size-1-i, 1) : set(size-1-i, 0);
      }
    }


};


int count(int N, uint32_t *A) {
  int count = 0;
  for(int i = 0; i < N; i++)
    count += bitcount(A[i]);
  return count;
}


int count2value(int bitnum, int N, uint32_t *A) {
  int c = count(N, A);
  return 2*c - bitnum;
}

void xnor_cpu(uint32_t complement, int N, uint32_t *A, uint32_t *B, uint32_t *C) {
  for(int i = 0; i < N; i++) 
    C[i] = ~(A[i]^B[i]);
  C[N-1] &= complement;
}

void xnor(Bitset *b1, Bitset *b2, Bitset *b3) {

  if(b1->bitnum != b2->bitnum || b1->bitnum != b3->bitnum) {
    printf("Bit num not equal");
  }
  int i = 0;

  xnor_cpu(b3->complement, b3->size, b1->bits, b2->bits, b3->bits);
}


int bin_dot(uint32_t offset, int N, uint32_t *A, uint32_t *B) {

  uint32_t *C = new uint32_t[N];
  int complement = (UINT32_MAX << (32 - offset)) >> (32-offset);
  xnor_cpu(complement, N, A, B, C);
  int val = count2value((N-1)*32+offset,N, C);
  delete C;
  return val;
}

//5x7 7x4 5x1 1x4
int bin_mat(uint32_t offset, int M, int N, int P, 
  float alpha, uint32_t *A, uint32_t *B, float *C) {

  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      C[i*N+j] = (float)bin_dot(offset, P, A+i*P, B+j*P);
      C[i*N+j] *= alpha;
    }
  }
}

void bin_gemm(int M, int N, int P,
  float alpha, float *A, float *B, float *C) {

  Bitset bA(M, P, A, false);
  Bitset bB(P, N, B, true);

   

  bin_mat(bA.offset, M, N, bB.alpha, bA.size, bA.bits, bB.bits, C);
}

