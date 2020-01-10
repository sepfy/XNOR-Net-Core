
using ::testing::ElementsAreArray;

TEST(BLAS, LINF_NORM) {
  int N = 3;
  int M = 2;
  float A[] = {-1, 3, 9, 1, -3, 2};
  float norm;
  norm = Linf_norm(N, M, A);
  EXPECT_EQ(9, norm);
}

TEST(BLAS, L1_NORM) {
  int N = 3;
  int M = 2;
  float A[] = {-1, 3, 9, 1, -3, 2};
  float norm;
  norm = L1_norm(N, M, A);
  EXPECT_EQ(19, norm);
}

TEST(BLAS, L2_NORM) {
  int N = 3;
  int M = 2;
  float A[] = {-1, 3, 9, 1, -3, 2};
  float norm;
  norm = L2_norm(N, M, A);
  EXPECT_EQ(105, norm);
}

 
TEST(BLAS, TRANSPOSE) { 

  int M = 3;
  int N = 2;
  float A[] = {3, 4, 1, 2, 1, 3}; //3x2
  float B[] = {3, 1, 1, 4, 2, 3}; //2x3
  float C[6] = {0}; //3x3
  transpose(M, N, A, C);

  ASSERT_THAT(C, ElementsAreArray(B));
}

