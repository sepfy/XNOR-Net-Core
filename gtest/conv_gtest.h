
using ::testing::ElementsAreArray;

TEST(LAYERS, CONVOLUTION_FORWARD) { 


  int W = 32, H = 32, C = 3;
  int FW = 5, FH = 5, FC = 20;
  int stride = 1, pad = 0;
  int batch = 100;

  float *inputs = new float[batch*W*H*C];
  for(int i = 0; i < batch*W*H*C; i++) {
    inputs[i] = 1.0;
  } 
  Convolution conv_test(W, H, C, FW, FH, FC, stride, pad);
  conv_test.batch = batch;
  conv_test.init();
  conv_test.input = inputs;


  for(int i = 0; i < FW*FH*C*FC; i++)
    conv_test.weight[i] = 1.0;
  for(int i = 0; i < FC; i++)
    conv_test.bias[i] = 0.0;

  conv_test.forward();
  int out_size = conv_test.batch*conv_test.out_w*conv_test.out_h*conv_test.FC;

  long long tmp = 0;

  for(int i = 0; i < out_size; i++) {
    tmp += (long long)conv_test.output[i];
  }

  long long solution = (long long)conv_test.batch
	            *(long long)conv_test.out_w
		    *(long long)conv_test.out_h
		    *(long long)conv_test.FC
		    *(long long)FW*(long long)FH*(long long)C;
  EXPECT_EQ(tmp, solution);
}

 
