#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "layers.h"
#include "gpu/util_gpu_gtest.h"
//#include "gpu/gemm_gpu_gtest.h"
//#include "gpu/blas_gpu_gtest.h"
#include "gpu/batchnorm_gpu_gtest.h"

using ::testing::ElementsAreArray;

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
