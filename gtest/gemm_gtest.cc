#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gemm.h"
#include "blas.h"
#include "utils.h"

using ::testing::ElementsAreArray;

 
TEST(BLAS, GEMM_TIME) { 

  int M = 400;
  int P = 19600;
  int N = 32;
  float *A = new float[M*P];
  float *B = new float[P*N];
  float *C = new float[M*N];
  float alpha = 1.0;

  for(int i = 0; i < 10000; i++) {
    ms_t start = getms();
    gemm_ta(M, N, P, alpha, A, B, C);
    printf("%llu ms\n", getms() - start);
  }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
