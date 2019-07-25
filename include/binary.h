#include <iostream>
#include <random>
#include <sys/time.h>
#include <math.h>
#include <stdint.h>
using namespace std;

class Bitset {

  public:
    uint64_t *bits;
    int bitnum;
    uint64_t complement;
    int offset;
    int N, BN;
    Bitset(){}; 
    ~Bitset(); 
    void init(int input_size);
    void set(float *inputs);
};


int bitcount(uint64_t n);
float popcount_xnor(Bitset *b1, Bitset *b2);
void bin_gemm(int M, int N, int P,
  float alpha, Bitset *bA, Bitset *bB, float *C);
void bin_gemm(int M, int N, int P,
  float alpha, float *A, float *B, float *C);

