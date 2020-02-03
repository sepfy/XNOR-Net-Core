
using ::testing::ElementsAreArray;

TEST(BLAS, COL_SUM) { 
  
  /*
  A = 0 1 3
      2 1 1
  */
  int M = 3;
  int N = 2;
  float *A = malloc_gpu(6);
  float *B = malloc_gpu(3);
  A[0] = 0, A[1] = 2, A[2] = 1, A[3] = 1, A[4] = 3, A[5] = 1;
  float *C = malloc_gpu(3);
  C[0] = 2, C[1] = 2, C[2] = 4; 
  col_sum_gpu(M, N, A, B);
  float err = 0.0;
  for(int i = 0; i < M*N; i++)
    if(B[i] != C[i]) {
      printf("%f\n", B[i]);
      err++;
    }
    
  EXPECT_EQ(err, 0.0);
}

