#include <iostream>
#include <random>
#include <sys/time.h>
#include <math.h>
#include <stdint.h>
using namespace std;

//typedef uint64_t BIT_BLK;
//#define BIT_BLK_MAX UINT64_MAX
typedef uint32_t BIT_BLK;
#define BIT_BLK_MAX UINT32_MAX

class Bitset {

  public:
    BIT_BLK *bits;
    int bitnum;
    BIT_BLK complement;
    int offset;
    int N, BN;
    Bitset(){}; 
    ~Bitset(); 
    void init(int input_size);
    void clean();
    void set(float *inputs);
};


int bitcount(BIT_BLK n);
float popcount_xnor(Bitset *b1, Bitset *b2);
void bin_gemm(int M, int N, int P,
  float alpha, Bitset *bA, Bitset *bB, float *C);
void bin_gemm(int M, int N, int P,
  float alpha, float *A, float *B, float *C);

