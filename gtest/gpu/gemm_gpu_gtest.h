using ::testing::ElementsAreArray;

 
TEST(BLAS, GEMM_GPU) {

  int M = 50176;
  int N = 20;
  int P = 75;



  float *A = malloc_gpu(M*P);
  float *B = malloc_gpu(P*N);
  float *C = malloc_gpu(M*N);
  float alpha = 1.0;
  gemm_gpu(TRS_N, TRS_N, 50176, 20, 75, alpha, A, B, C);

  float *D = new float[M*N];
  gpu_pull_array(C, D, M*N);

  long long tmp = 0;
  for(int i = 0; i < M*N; i++)
    tmp += (long)0.0;

  long long solution = (long long)M*P*N;
  EXPECT_EQ(tmp, solution);
}
/*
TEST(BLAS, GEMM_GPU_2) {

  int M = 5;
  int P = 3;
  int N = 5;
  float *A = malloc_gpu(M*P);
  float *B = malloc_gpu(P*N);
  float *C = malloc_gpu(M*N);
  float alpha = 1.0;


  for(int i = 0; i < M*P; i++)
    random_normal(M*P, A);

  for(int i = 0; i < P*N; i++)
    random_normal(N*P, B);

  gemm_gpu(TRS_N, TRS_N, M, N, P, alpha, A, B, C);
  float tmp1 = 0.0;
  for(int i = 0; i < M*N; i++) {
    tmp1 += C[i];
  }

  gemm(M, N, P, alpha, A, B, C);

  float tmp2 = 0.0;
  for(int i = 0; i < M*N; i++) {
    tmp2 += C[i];
  }

  EXPECT_EQ(round(tmp1, 5), round(tmp2, 5));
}
*/
TEST(GEMM, SGEMM_TRS_A) {

  int M = 30;
  int P = 20;
  int N = 50;
  // A = P*M, B = N*P, M*N
  float *A = malloc_gpu(M*P);
  float *B = malloc_gpu(P*N);
  float *C = malloc_gpu(M*N);
  float alpha = 1.0;

  for(int i = 0; i < M*P; i++)
    A[i] = 1.0;

  for(int i = 0; i < P*N; i++)
    B[i] = 1.0;

  gemm_gpu(TRS_T, TRS_N, M, N, P, alpha, A, B, C);

  long long tmp = 0;
  for(int i = 0; i < M*N; i++)
    tmp += (long)C[i];

  long long solution = (long long)M*(long long)P*(long long)N;
  EXPECT_EQ(tmp, solution);
}

TEST(GEMM, SGEMM_TRS_B) {

  int M = 30;
  int P = 20;
  int N = 50;
  // A = M*P, B = N*P, M*N
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

  gemm_gpu(TRS_N, TRS_T, M, N, P, alpha, A, B, C);

  long long tmp = 0;
  for(int i = 0; i < M*N; i++)
    tmp += (long)C[i];

  long long solution = (long long)M*(long long)P*(long long)N;
  EXPECT_EQ(tmp, solution);
}



