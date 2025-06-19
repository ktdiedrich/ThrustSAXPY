#include <gtest/gtest.h>
#include <cudnn.h>

TEST(CuDNNTest, CanCreateHandle) {
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle);
    EXPECT_EQ(status, CUDNN_STATUS_SUCCESS) << "cuDNN handle creation failed: " << cudnnGetErrorString(status);
    if (status == CUDNN_STATUS_SUCCESS) {
        cudnnDestroy(handle);
    }
}

TEST(CuDNNTest, VersionIsNonZero) {
    EXPECT_GT(CUDNN_VERSION, 0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}