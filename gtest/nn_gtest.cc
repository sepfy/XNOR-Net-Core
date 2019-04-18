#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gemm.h"

using ::testing::ElementsAreArray;
 
TEST(BlasTest, GEMM) { 

  int M = 3;
  int N = 2;
  int P = 3;
  float A[] = { 3, 4, 1, 2, 1, 3, 2, 2, 1}; //3x3
  float B[] = { 1, 2, 3, 2, 1, 2};  //3x2
  float D[] = {32, 32, 16, 24, 18, 20}; //3x2
  float alpha = 2;
  float C[6] = {0};
  gemm(M, N, P, alpha, A, B, C);

  ASSERT_THAT(C, ElementsAreArray(D));

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
