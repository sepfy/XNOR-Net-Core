using ::testing::ElementsAreArray;

 
TEST(GEMM, GEMM) { 

  int M = 7840;
  int P = 760;
  int N = 200;
  float *A = new float[M*P];
  float *B = new float[P*N];
  float *C = new float[M*N];
  float alpha = 1.0;

  for(int i = 0; i < M*P; i++)
    A[i] = 1.0;

  for(int i = 0; i < P*N; i++)
    B[i] = 1.0;

  for(int i = 0; i < M*N; i++)
    C[i] = 1.0;

  ms_t s = getms();
  gemm(M, N, P, alpha, A, B, C);
  printf("gemm time = %lld\n", getms() - s);

  long long tmp = 0;
  for(int i = 0; i < M*N; i++)
    tmp += (long)C[i];

  long long solution = (long long)M*(long long)P*(long long)N;
  EXPECT_EQ(tmp, solution);
}

