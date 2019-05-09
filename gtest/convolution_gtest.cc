#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gemm.h"
#include "blas.h"
#include "utils.h"
#include "layers.h"

using ::testing::ElementsAreArray;

TEST(CONVOLUTION, FORWARD1) {

/*
  (N, H, W, C) = (1, 4, 4, 2)  
  filter = (3, 3, 3), stride = 1 padding = 0
  
  Weight:
    1 1 1  2 2 2
    1 1 1  2 2 2
    1 1 1  2 2 2

    3 3 3  0 0 0
    3 3 3  0 0 0
    3 3 3  0 0 0

    1 2 1  6 6 6
    2 1 2  0 0 0
    1 2 1  1 1 1      
  Input:                  Output:
    1 2 1 4  3 4 2 2        98  94   66  78   113  102
    3 4 2 2  1 2 2 3       102 110   90 108    81  102
    1 6 2 3  9 8 7 4        
    3 7 2 8  1 2 4 5       
*/ 
  //18*3
  float weight[] = {1,3,1,2,0,6,1,3,2,2,0,6,1,3,1,2,0,6,
                    1,3,2,2,0,0,1,3,1,2,0,0,1,3,2,2,0,0,
                    1,3,1,2,0,1,1,3,2,2,0,1,1,3,1,2,0,1};


  float input[] = {1,3,2,4,1,2,4,2,
                   3,1,4,2,2,2,2,3,
                   1,9,6,8,2,7,3,4,
                   3,1,7,2,2,4,8,5};

  float output[] = {98,66,113,94,78,102,
                    102,90,81,110,108,102};
/*
  float weight[] = {1,3,1,
                    1,3,2,
                    1,3,1,
                    1,3,2,
                    1,3,1,
                    1,3,2,
                    1,3,1,
                    2,0,6,
                    2,0,6,
                    2,0,6,
                    2,0,0,
                    2,0,0,
                    2,0,0,
                    2,0,1,
                    2,0,1,
                    2,0,1};
*/                 
                    
  //   4*18  18x3

  int W = 4, H = 4, C = 2;
  int FW = 3, FH = 3, FC = 3;
  int stride = 1, pad = 0;

  float conv_output[12] = {0};

  Convolution conv(1, 4, 4, 2, 3, 3, 3, 1, 0, input);
  //conv.weight = weight;
  memcpy(conv.weight, weight, 54*sizeof(float));
  
  conv.forward();
  memcpy(conv_output, conv.output, 12*sizeof(float));
  ASSERT_THAT(conv_output, ElementsAreArray(output));
}

int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
