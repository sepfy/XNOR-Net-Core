#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "layers.h"

#include "gemm_gtest.h"
#include "util_gtest.h"
#include "conv_gtest.h"
#include "blas_gtest.h"
#include "pooling_gtest.h"
using ::testing::ElementsAreArray;

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
