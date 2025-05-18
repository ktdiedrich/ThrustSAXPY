#include "image_loader.h"
#include <cnpy.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <vector>
#include <stdexcept>
#include <iostream>


ImageLoader::ImageLoader() 
{
}

std::map<std::string, std::vector<thrust::device_vector<float>>> ImageLoader::read_all_npz_arrays(const std::string& data_file) {
    std::map<std::string, std::vector<thrust::device_vector<float>>> all_arrays;

    // Load the NPZ file
    cnpy::npz_t npz = cnpy::npz_load(data_file);

    for (const auto& kv : npz) {
        const std::string& array_name = kv.first;
        cnpy::NpyArray arr = kv.second;

        // Only handle float arrays with 2 dimensions
        if (arr.word_size != sizeof(float) || arr.shape.size() != 2) {
            std::cerr << "Skipping array " << array_name << " (not 2D float)" << std::endl;
            continue;
        }

        float* data = arr.data<float>();
        size_t num_rows = arr.shape[0];
        size_t row_size = arr.shape[1];

        std::vector<thrust::device_vector<float>> vectors;
        for (size_t i = 0; i < num_rows; ++i) {
            thrust::device_vector<float> vec(data + i * row_size, data + (i + 1) * row_size);
            vectors.push_back(std::move(vec));
        }
        all_arrays[array_name] = std::move(vectors);
    }
    return all_arrays;
}
