#pragma once

#include <string>
#include <vector>
#include <cnpy.h>
#include <map>


/** @brief Load data from compressed files.
 * @param data_file Path to the NPZ file.
 * @return A vector of device vectors containing the loaded data.
 */
std::map<std::string, cnpy::NpyArray> read_all_npz_arrays(const std::string& data_file);

template<typename T>
void load_array_to_vectors(const cnpy::NpyArray& array,
    std::vector<std::vector<T>>& vec2d,
    std::vector<std::vector<std::vector<T>>>& vec3d)
{
    const uint8_t* data = array.data<T>();
    if (!data) {
        throw std::runtime_error("Array data is null.");
    }
    if (array.shape.size() == 2) {
        size_t rows = array.shape[0];
        size_t cols = array.shape[1];
        size_t total = rows * cols;
        
        if (array.num_vals != total) {
            throw std::runtime_error("Array size mismatch: expected " +
                std::to_string(total) + ", got " + std::to_string(array.num_vals));
        }
        vec2d.resize(rows, std::vector<T>(cols));
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                try {
                    vec2d[i][j] = data[i * cols + j];
                } catch (const std::exception& e) {
                    throw std::runtime_error("Error accessing 2D array data: " + std::string(e.what()));
                }
        return;
    } else if (array.shape.size() == 3) {
        size_t slices = array.shape[0];
        size_t rows = array.shape[1];
        size_t cols = array.shape[2];
        size_t total = slices * rows * cols;
        
        if (array.num_vals != total) {
            throw std::runtime_error("Array size mismatch: expected " +
                std::to_string(total) + ", got " + std::to_string(array.num_vals));
        }
        vec3d.resize(slices, std::vector<std::vector<T>>(rows, std::vector<T>(cols)));
        for (size_t s = 0; s < slices; ++s)
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    try {
                        vec3d[s][i][j] = data[s * rows * cols + i * cols + j];
                    } catch (const std::exception& e) {
                        throw std::runtime_error("Error accessing 3D array data: " + std::string(e.what()));
                    }
    } else {
        throw std::runtime_error("Only 3D arrays are supported.");
    }
}

