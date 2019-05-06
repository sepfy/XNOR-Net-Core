#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gemm.h"
#include "blas.h"
#include "utils.h"
#include "layers.h"

using ::testing::ElementsAreArray;

TEST(LAYERS, POOLING1) {
/*
  (N, H, W, C) = (1, 4, 4, 2)  
  filter = (2, 2, 2), stride = 2 padding = 0
   
  Input:                  Output:
    1 2 1 4  3 4 2 2        4 4  4 3 
    3 4 2 2  1 2 2 3        7 8  9 7
    1 6 2 3  9 8 7 4        
    3 7 2 8  1 2 4 5       
*/

  float input[] = {1, 3, 2, 4, 1, 2, 4, 2,
                   3, 1, 4, 2, 2, 2, 2, 3,
                   1, 9, 6, 8, 2, 7, 3, 4,
                   3, 1, 7, 2, 2, 4, 8, 5};

  float output[] = {4, 4, 4, 3,
                    7, 9, 8, 7};

  int W = 4, H = 4, C = 2;
  int FW = 2, FH = 2, FC = 2;
  int stride = 2, pad = 0;

  float pooling_output[8] = {0};

  Pooling pool(1, 4, 4, 2, 2, 2, 2, 2, 0, input);
  pool.forward();
  memcpy(pooling_output, pool.output, 8*sizeof(float));
  ASSERT_THAT(pooling_output, ElementsAreArray(output));
}

TEST(LAYERS, POOLING2) {
/*
  (N, H, W, C) = (1, 4, 4, 2)  
  filter = (2, 2, 2), stride = 2 padding = 0
   
  Input:                          Output:
    1 2 1 4  3 4 2 2  3 3 6 6       4 4  4 3  3 6
    3 4 2 2  1 2 2 3  2 2 3 1       7 8  9 7  8 8 
    1 6 2 3  9 8 7 4  7 8 8 7       
    3 7 2 8  1 2 4 5  5 5 6 6   

    1 2 1 5  3 4 2 2  3 3 6 6       4 5  4 4  5 6
    3 4 2 2  1 2 4 3  2 5 3 1       7 9  2 7  8 6
    1 6 9 3  1 1 7 4  7 8 1 1       
    3 7 2 8  1 2 4 5  5 5 6 6   

*/

  float input[] = {1, 3, 3, 2, 4, 3, 1, 2, 6, 4, 2, 6,
                   3, 1, 2, 4, 2, 2, 2, 2, 3, 2, 3, 1,
                   1, 9, 7, 6, 8, 8, 2, 7, 8, 3, 4, 7,
                   3, 1, 5, 7, 2, 5, 2, 4, 6, 8, 5, 6,
                   1, 3, 3, 2, 4, 3, 1, 2, 6, 5, 2, 6,
                   3, 1, 2, 4, 2, 5, 2, 4, 3, 2, 3, 1,
                   1, 1, 7, 6, 1, 8, 9, 7, 1, 3, 4, 1,
                   3, 1, 5, 7, 2, 5, 2, 4, 6, 8, 5, 6};

  float output[] = {4, 4, 3, 4, 3, 6,
                    7, 9, 8, 8, 7, 8,
                    4, 4, 5, 5, 4, 6,
                    7, 2, 8, 9, 7, 6};

  int W = 4, H = 4, C = 3;
  int FW = 2, FH = 2, FC = 3;
  int stride = 2, pad = 0;

  float pooling_output[24] = {0};

  Pooling pool(2, 4, 4, 3, 2, 2, 3, 2, 0, input);
  pool.forward();
  memcpy(pooling_output, pool.output, 24*sizeof(float));
  ASSERT_THAT(pooling_output, ElementsAreArray(output));
}

int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
