using ::testing::ElementsAreArray;

 
TEST(BLAS, GEMM_GPU) {

  int N = 12;
  float *x_gpu = malloc_gpu(N);

  float x[N] = {0};
  for(int i = 0; i < N; i++)
      x[i] = (float)i;

  gpu_push_array(x_gpu, x, N);

  Batchnorm bn1(4);
  bn1.batch = 3;
  bn1.init();
  bn1.input = x_gpu;
  bn1.forward();

  gpu_pull_array(bn1.output, x, N);
  
  for(int i = 0; i < N; i++)
    cout << x[i] << " ";
  cout << endl;

  float tmp = 0.0;
  float solution = 0.0;

  EXPECT_EQ(tmp, solution);
}


