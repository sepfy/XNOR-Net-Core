
using ::testing::ElementsAreArray;

TEST(UTILITY, IM2COL) {

  /*
   *  (N, H, W, C) = (1, 3, 3, 2)  
   *  filter = (2, 2, 2), stride = 1 padding = 1
   *
   *  Input:                   Output:
   *    1 2 1   3 4 2            1 3 2 4 3 1 4 2
   *    3 4 2   1 2 2            2 4 1 2 4 2 2 2
   *    1 6 2   9 8 7            3 1 4 2 1 9 6 8
   *                             4 2 2 2 6 8 2 7
   *
   */

  //CPU
  float *im = new float[18];
  im[0] = 1, im[1] = 3, im[2] = 2, im[3] = 4, im[4] = 1, im[5] = 2;
  im[6] = 3, im[7] = 1, im[8] = 4, im[9] = 2, im[10] = 2, im[11] = 2;
  im[12] = 1, im[13] = 9, im[14] = 6, im[15] = 8, im[16] = 2, im[17] = 7;
  
  float *im_gpu = malloc_gpu(18);
  gpu_push_array(im_gpu, im, 18);
  
  float col[] = {1, 3, 2, 4, 3, 1, 4, 2,
                 2, 4, 1, 2, 4, 2, 2, 2,
                 3, 1, 4, 2, 1, 9, 6, 8,
                 4, 2, 2, 2, 6, 8, 2, 7};

  int W = 3, H = 3, C = 2;
  int FW = 2, FH = 2, FC = 2;
  int stride = 1, pad = 0;
  int out_w = (W + 2*pad - FW)/stride + 1; 
  int out_h = (H + 2*pad - FH)/stride + 1; 
  int channel_out = FW*FH*C;

  float *col_out_gpu = malloc_gpu(out_w*out_h*channel_out);

  im2col_gpu(W, H, C, FW, FH, FC,
             stride, pad, im_gpu, col_out_gpu);


  float col_out[32];
  gpu_pull_array(col_out_gpu, col_out, 32);

  int err = 0;
  //printf("%d\n", out_w*out_h*channel_out);
  for(int i = 0; i < out_w*out_h*channel_out; i++)
    if(col_out[i] != col[i]) {
      err++;
      printf("col_out[%d] = %f != %f = col[%d]\n", i, col_out[i], col[i], i);
    }

  EXPECT_EQ(0, err);
}


TEST(UTILITY, IM2COL_PADDING) {

  /*
   *  Input:                    
   *    0 0 0 0 0  0 0 0 0 0 
   *    0 1 2 1 0  0 3 4 2 0
   *    0 3 4 2 0  0 1 2 2 0 
   *    0 1 6 2 0  0 9 8 7 0
   *    0 0 0 0 0  0 0 0 0 0
   */

  float col[] = {0,0,0,0,0,0,0,0,1,3,2,4,0,0,3,1,4,2,
	         0,0,0,0,0,0,1,3,2,4,1,2,3,1,4,2,2,2,
		 0,0,0,0,0,0,2,4,1,2,0,0,4,2,2,2,0,0,
		 0,0,1,3,2,4,0,0,3,1,4,2,0,0,1,9,6,8,
		 1,3,2,4,1,2,3,1,4,2,2,2,1,9,6,8,2,7,
		 2,4,1,2,0,0,4,2,2,2,0,0,6,8,2,7,0,0,
		 0,0,3,1,4,2,0,0,1,9,6,8,0,0,0,0,0,0,
		 3,1,4,2,2,2,1,9,6,8,2,7,0,0,0,0,0,0,
		 4,2,2,2,0,0,6,8,2,7,0,0,0,0,0,0,0,0};

  //CPU
  float *im = new float[18];
  im[0] = 1, im[1] = 3, im[2] = 2, im[3] = 4, im[4] = 1, im[5] = 2;
  im[6] = 3, im[7] = 1, im[8] = 4, im[9] = 2, im[10] = 2, im[11] = 2;
  im[12] = 1, im[13] = 9, im[14] = 6, im[15] = 8, im[16] = 2, im[17] = 7;
  
  float *im_gpu = malloc_gpu(18);
  gpu_push_array(im_gpu, im, 18);
  

  int W = 3, H = 3, C = 2;
  int FW = 3, FH = 3, FC = 2;
  int stride = 1, pad = 1;
  int out_w = (W + 2*pad - FW)/stride + 1; 
  int out_h = (H + 2*pad - FH)/stride + 1; 
  int channel_out = FW*FH*C;

  int N = out_w*out_h*channel_out;
  float *col_out_gpu = malloc_gpu(N);

  im2col_gpu(W, H, C, FW, FH, FC,
             stride, pad, im_gpu, col_out_gpu);

  float col_out[N];
  gpu_pull_array(col_out_gpu, col_out, N);

  int err = 0;
  //printf("%d\n", out_w*out_h*channel_out);
  for(int i = 0; i < N; i++)
    if(col_out[i] != col[i]) {
      err++;
      printf("col_out[%d] = %f != %f = col[%d]\n", i, col_out[i], col[i], i);
    }

  EXPECT_EQ(0, err);
}



#if 0
TEST(UTILITY, COL2IM) {

  /*
   *  (N, H, W, C) = (1, 3, 3, 2)  
   *  filter = (2, 2, 2), stride = 1 padding = 1
   *
   *  Output:                  Input:
   *    1 2 1   3 4 2            1 3 2 4 3 1 4 2
   *    3 4 2   1 2 2            2 4 1 2 4 2 2 2
   *    1 6 2   9 8 7            3 1 4 2 1 9 6 8
   *                             4 2 2 2 6 8 2 7
   *
   */

  float im[] = {1, 3, 2, 4, 1, 2, 3, 1, 4, 2, 2, 2, 1, 9, 6, 8, 2, 7};
  float col[] = {1, 3, 2, 4, 3, 1, 4, 2,
                 2, 4, 1, 2, 4, 2, 2, 2,
                 3, 1, 4, 2, 1, 9, 6, 8,
                 4, 2, 2, 2, 6, 8, 2, 7};

  int W = 3, H = 3, C = 2;
  int FW = 2, FH = 2, FC = 2;
  int stride = 1, pad = 0;

  int out_w = (W + 2*pad - FW)/stride + 1;  //3-2+1 = 2
  int out_h = (H + 2*pad - FH)/stride + 1;  //3-2+1 = 2
  int channel_out = FW*FH*C;

  //float *col_out = new float[out_w*out_h*channel_out];
  float im_out[18] = {0};
  col2im(W, H, C, FW, FH, FC,
             stride, pad, im_out, col);

  ASSERT_THAT(im_out, ElementsAreArray(im));
}
#endif
#if 0
TEST(UTILITY, COL2IM) {

  /*
   *  (N, H, W, C) = (1, 3, 3, 1)  
   *  filter = (2, 2, 2), stride = 1 padding = 1
   *
   *  Output:                  Input:
   *    1 2 1   3 4 2            1 3 2 4 3 1 4 2
   *    3 4 2   1 2 2            2 4 1 2 4 2 2 2
   *    1 6 2   9 8 7            3 1 4 2 1 9 6 8
   *                             4 2 2 2 6 8 2 7
   *
   */

  float im[] = {1, 3, 2, 4, 1, 2, 3, 1, 4, 2, 2, 2, 1, 9, 6, 8, 2, 7};
  float col[] = {1, 3, 2, 4, 3, 1, 4, 2,
                 2, 4, 1, 2, 4, 2, 2, 2,
                 3, 1, 4, 2, 1, 9, 6, 8,
                 4, 2, 2, 2, 6, 8, 2, 7};

  int W = 3, H = 3, C = 2;
  int FW = 2, FH = 2, FC = 2;
  int stride = 1, pad = 0;

  int out_w = (W + 2*pad - FW)/stride + 1;  //3-2+1 = 2
  int out_h = (H + 2*pad - FH)/stride + 1;  //3-2+1 = 2
  int channel_out = FW*FH*C;

  //float *col_out = new float[out_w*out_h*channel_out];
  float im_out[18] = {0};;
  col2im(W, H, C, FW, FH, FC,
             stride, pad, im_out, col);

  ASSERT_THAT(im_out, ElementsAreArray(im));
}

#endif
