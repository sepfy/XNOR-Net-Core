
using ::testing::ElementsAreArray;

TEST(LAYERS, POOLING) {

  int W = 32, H = 32, C = 3;
  int FW = 2, FH = 2, FC = 3;
  int stride = 2, pad = 0;
  int batch = 100;

  float *inputs = new float[batch*W*H*C];
  for(int i = 0; i < batch*W*H*C; i++) {
    inputs[i] = 1.0;
  }
  Pooling pool(W, H, C, FW, FH, FC, stride, pad);
  pool.batch = batch;
  pool.init();
  pool.input = inputs;
  pool.forward();

  int out_size = pool.batch*pool.out_w*pool.out_h*pool.FC;

  long long tmp = 0;

  for(int i = 0; i < out_size; i++) {
    tmp += (long long)pool.output[i];
  }

  long long solution = (long long)pool.batch
                    *(long long)pool.out_w
                    *(long long)pool.out_h
                    *(long long)C;
  EXPECT_EQ(tmp, solution);

}


