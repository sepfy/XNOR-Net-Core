#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "layers.h"

#include "cpu/gemm_gtest.h"
#include "cpu/util_gtest.h"
#include "cpu/conv_gtest.h"
#include "cpu/blas_gtest.h"
#include "cpu/pooling_gtest.h"

using ::testing::ElementsAreArray;

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
