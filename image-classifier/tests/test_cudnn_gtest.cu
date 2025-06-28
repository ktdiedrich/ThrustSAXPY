#include <gtest/gtest.h>
#include <cudnn.h>
#include "minimal.h"
#include "graph.h"


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

TEST(CuDNNTest, SigmoidActivationRuns) {
    // This will run your minimal sigmoid activation example.
    // If it doesn't throw or crash, the test passes.
    sigmoid_activate_tensor();
    SUCCEED();
}


TEST(CuDNNTest, GraphCreation) {
    // Test if the graph can be created without errors.
    create_graph();
    SUCCEED();
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}