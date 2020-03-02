
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
//16, 31 15 25
//   
// 17 -> (89  ,105, 137, 162
//       (90  ,106, 138, 163
// i = 3, j = 3  k = 2 =>

/*
  (i+pad)*W + (j+pad) = 3*3+3 = 12

  index/W

  

  (row + pad)*

   im_row = offset_h + i*stride;
  im_col = offset_w + j*stride;

  int col_idx = (i*out_w + j)*out_col + k;
  int im_pad_row = im_row - pad;
  int im_pad_col = im_col - pad;
 

  18*5 - 1 = 89
  18*6 - 3 = 105
  18*7 - 5 
  18*8 - 7 = 137
  18*9 - 9 = 162

  1, 3, 5, 7, 9, 11, 13, 15, 17

  w = 3 , h = 3


  3*2


  for(i = 0; i < FH; i++)
    for(j = 0; j < FW; j++) {
      int c = (i*FW + j)*C + k;
      int col_idex = out_col*? - c
    }
  }


*/


// offset_w = 2
// offset_h = 2
// 
//   k = ?*C + c_im
//   k = C*(?*FW + offset_w)
//   k = offset*FW*C
//    c_im = k % C;
//    offset_w = k / C % FW;
//    offset_h = k / C / FW;
//             = (k~C)/(C~FW)/FW
//18*5 = 
//18*4 + 17= 89
//18*5 + 17 = 107
// offset_h = k/C/FW
// 2 = k/2/3
// 6, 7, 8 = k/2 
// k = 12, 13, 14, 15, 16, 17
// 
//
// out_col = 3*3*2 = 18 
// offset_w = k/C%FW
// 2 = k/2%3
// a*3 + 2 = k/2
// k/2 = 8 k = 16, 17
// k => 17
// 15  1 2



 
// W=H=3 C=2 out_col = 18
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



TEST(UTILITY, COL2IM_PADDING) {
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

  float *im = new float[18];
  im[0] = 4, im[1] = 12, im[2] = 12, im[3] = 24, im[4] = 4, im[5] = 8;
  im[6] = 18, im[7] = 6, im[8] = 36, im[9] = 18, im[10] = 12, im[11] = 12;
  im[12] = 4, im[13] = 36, im[14] = 36, im[15] = 48, im[16] = 8, im[17] = 28;
  int W = 3, H = 3, C = 2;
  int FW = 3, FH = 3, FC = 2;
  int stride = 1, pad = 1;
  int out_w = (W + 2*pad - FW)/stride + 1; 
  int out_h = (H + 2*pad - FH)/stride + 1; 
  int channel_out = FW*FH*C;

  int N = out_w*out_h*channel_out;
  float *col_gpu = malloc_gpu(N);

  gpu_push_array(col_gpu, col, N);

  
  float *im_out_gpu = malloc_gpu(18);
  float im_out[18];

  col2im_gpu(W, H, C, FW, FH, FC,
             stride, pad, im_out_gpu, col_gpu);
  gpu_pull_array(im_out_gpu, im_out, 18);

  int err = 0;
  //printf("%d\n", out_w*out_h*channel_out);
  for(int i = 0; i < 18; i++)
    if(im_out[i] != im[i]) {
      err++;
      printf("im_out[%d] = %f != %f = im[%d]\n", i, im_out[i], im[i], i);
    }

  EXPECT_EQ(0, err);
}

TEST(UTILITY, COL2IM_PADDING_1) {
  /*
   *  (N, H, W, C) = (1, 4, 4, 3)  
   *  filter = (3, 3, 3), stride = 1 padding = 1
   *
   *  Input:                 
   *   0 0 0 0 0 0  0 0 0 0 0 0  0 0 0 0 0 0
   *   0 3 1 2 1 0  0 3 4 2 5 0  0 2 3 4 3 0    
   *   0 2 3 4 2 0  0 1 2 2 3 0  0 1 1 2 3 0
   *   0 1 1 6 2 0  0 9 8 7 1 0  0 3 2 3 5 0
   *   0 0 1 2 7 0  0 6 6 2 1 0  0 5 6 7 8 0
   *   0 0 0 0 0 0  0 0 0 0 0 0  0 0 0 0 0 0  
   */
  float im[] = {12,12,8,6,24,18,12,12,24,4,20,12,
                12,6,6,27,18,9,36,18,18,12,18,18,
                6,54,18,9,72,18,54,63,27,12,6,30,
                0,24,20,6,36,36,12,12,42,28,4,32};
  float col[] = {0,0,0,0,0,0,0,0,0,0,0,0,3,3,2,1,4,3,0,0,0,2,1,1,3,2,1,
                 0,0,0,0,0,0,0,0,0,3,3,2,1,4,3,2,2,4,2,1,1,3,2,1,4,2,2,
                 0,0,0,0,0,0,0,0,0,1,4,3,2,2,4,1,5,3,3,2,1,4,2,2,2,3,3,
                 0,0,0,0,0,0,0,0,0,2,2,4,1,5,3,0,0,0,4,2,2,2,3,3,0,0,0,
                 0,0,0,3,3,2,1,4,3,0,0,0,2,1,1,3,2,1,0,0,0,1,9,3,1,8,2,
                 3,3,2,1,4,3,2,2,4,2,1,1,3,2,1,4,2,2,1,9,3,1,8,2,6,7,3,
                 1,4,3,2,2,4,1,5,3,3,2,1,4,2,2,2,3,3,1,8,2,6,7,3,2,1,5,
                 2,2,4,1,5,3,0,0,0,4,2,2,2,3,3,0,0,0,6,7,3,2,1,5,0,0,0,
                 0,0,0,2,1,1,3,2,1,0,0,0,1,9,3,1,8,2,0,0,0,0,6,5,1,6,6,
                 2,1,1,3,2,1,4,2,2,1,9,3,1,8,2,6,7,3,0,6,5,1,6,6,2,2,7,
                 3,2,1,4,2,2,2,3,3,1,8,2,6,7,3,2,1,5,1,6,6,2,2,7,7,1,8,
                 4,2,2,2,3,3,0,0,0,6,7,3,2,1,5,0,0,0,2,2,7,7,1,8,0,0,0,
                 0,0,0,1,9,3,1,8,2,0,0,0,0,6,5,1,6,6,0,0,0,0,0,0,0,0,0,
                 1,9,3,1,8,2,6,7,3,0,6,5,1,6,6,2,2,7,0,0,0,0,0,0,0,0,0,
                 1,8,2,6,7,3,2,1,5,1,6,6,2,2,7,7,1,8,0,0,0,0,0,0,0,0,0,
                 6,7,3,2,1,5,0,0,0,2,2,7,7,1,8,0,0,0,0,0,0,0,0,0,0,0,0};

  int W = 4, H = 4, C = 3;
  int FW = 3, FH = 3, FC = 2;
  int stride = 1, pad = 1;
  int out_w = (W + 2*pad - FW)/stride + 1; 
  int out_h = (H + 2*pad - FH)/stride + 1; 
  int channel_out = FW*FH*C;

  int N = out_w*out_h*channel_out;
  float *col_gpu = malloc_gpu(N);

  gpu_push_array(col_gpu, col, N);

  
  float *im_out_gpu = malloc_gpu(48);
  float im_out[48];

  col2im_gpu(W, H, C, FW, FH, FC,
             stride, pad, im_out_gpu, col_gpu);
  gpu_pull_array(im_out_gpu, im_out, 48);

  int err = 0;
  for(int i = 0; i < 48; i++)
    if(im_out[i] != im[i]) {
      err++;
      printf("im_out[%d] = %f != %f = im[%d]\n", i, im_out[i], im[i], i);
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
