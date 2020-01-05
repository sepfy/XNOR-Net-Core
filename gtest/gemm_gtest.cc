#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gemm.h"
#include "blas.h"
#include "utils.h"

using ::testing::ElementsAreArray;

 
TEST(BLAS, GEMM_TIME) { 

  int M = 7840;
  int P = 750;
  int N = 200;
  float *A = new float[M*P];
  float *B = new float[P*N];
  float *C = new float[M*N];
  float alpha = 1.0;

  gemm(M, N, P, alpha, A, B, C);

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
