#include "graph.h"

#include <cudnn_frontend.h>
#include <memory>
#include <array>


void create_graph() {
    auto tensor = cudnn_frontend::TensorBuilder()
        .setDim(4, std::array<int64_t, 4>{1, 3, 28, 28}.data()) // Example dimensions for an image tensor
        .setDataType(CUDNN_DATA_FLOAT)
        .setAlignment(16)
        .setId(0)
        .setStrides(4, std::array<int64_t, 4>{3 * 28 * 28, 28 * 28, 28, 1}.data()) // Example strides
        .build();
}
