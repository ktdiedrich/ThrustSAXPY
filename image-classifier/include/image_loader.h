#pragma once

#include <string>
#include <vector>
#include <cnpy.h>
#include <map>
#include <opencv2/opencv.hpp>


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


template<typename DataType>
void write_image(const std::string& array_name,
                 const std::vector<std::vector<DataType>>& vec2d, const int slice_numer,
                 const int cv_type = CV_8UC1,
                 const std::string encoding = "png") {
    // TODO: CV_8UC1 is  macro for 8-bit single channel image
    cv::Mat img(vec2d.size(), vec2d[0].size(), cv_type);
    for (size_t i = 0; i < vec2d.size(); ++i)
        for (size_t j = 0; j < vec2d[0].size(); ++j)
            img.at<DataType>(i, j) = vec2d[i][j];
    std::string filename = array_name + "_slice" + std::to_string(slice_numer) + "." + encoding;
    if (encoding == "jpg" || encoding == "jpeg") {
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 95 };
        cv::imwrite(filename, img, params);
    } else if (encoding == "png") {
        std::vector<int> params = { cv::IMWRITE_PNG_COMPRESSION, 9 };
        cv::imwrite(filename, img, params);
    } else {
        throw std::runtime_error("Unsupported image encoding: " + encoding);
    }
    std::cout << " Wrote: " << filename;
}


/** Calculate the histogram of one-hot encoded vectors in vec2d.
 * Each row in vec2d is a one-hot vector; the histogram counts the index of the 'hot' (nonzero) entry. */ 
template<typename T>
std::map<size_t, size_t> one_hot_histogram(const std::vector<std::vector<T>>& vec2d)
{
    std::map<size_t, size_t> histogram;
    for (const auto& row : vec2d) {
        auto it = std::find_if(row.begin(), row.end(), [](T v) { return v != T(0); });
        if (it != row.end()) {
            size_t idx = std::distance(row.begin(), it);
            ++histogram[idx];
        }
    }
    return histogram;
}
