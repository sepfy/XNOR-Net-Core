#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gemm.h"
#include "blas.h"
#include "utils.h"
#include "gpu.h"

using ::testing::ElementsAreArray;

 
TEST(BLAS, GEMM_GPU) { 

  int M = 7840;
  int P = 760;
  int N = 200;
  float *A = malloc_gpu(M*P);
  float *B = malloc_gpu(P*N);
  float *C = malloc_gpu(M*N);
  float alpha = 1.0;

  for(int i = 0; i < M*P; i++)
    A[i] = 1.0;

  for(int i = 0; i < P*N; i++)
    B[i] = 1.0;

  for(int i = 0; i < M*N; i++)
    C[i] = 1.0;

  ms_t s = getms();
  gemm_gpu(M, N, P, alpha, A, B, C);
  printf("gemm time = %lld\n", getms() - s);

  long long tmp = 0;
  for(int i = 0; i < M*N; i++)
    tmp += (long)C[i];

  long long solution = (long long)M*P*N;
  EXPECT_EQ(tmp, solution);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
