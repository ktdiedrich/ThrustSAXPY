#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "Error: No CUDA-capable devices found." << std::endl;
        return 1;
    }

    // Get properties of the first device
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Compute capability
    std::cout << deviceProp.major << deviceProp.minor << std::endl;

    return 0;
}