#pragma once

#include <string>
#include <vector>
#include <cnpy.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


const std::map<size_t, std::string> CHEST_LABELS = {
    {0, "atelectasis"}, {1, "cardiomegaly"}, {2, "effusion"}, {3, "infiltration"}, {4, "mass"}, {5, "nodule"},
    {6, "pneumonia"}, {7, "pneumothorax"}, {8, "consolidation"}, {9, "edema"}, {10, "emphysema"},
    {11, "fibrosis"}, {12, "pleural"}, {13, "hernia"}
};


/** @brief Load data from compressed files.
 * @param data_file Path to the NPZ file.
 * @return A vector of device vectors containing the loaded data.
 */
std::map<std::string, cnpy::NpyArray> read_all_npz_arrays(const std::string& data_file);

// 2D version: generic container type (default is std::vector)
template<
    typename T,
    template <typename, typename> class Container = std::vector,
    typename AllocOuter = std::allocator<Container<T, std::allocator<T>>>,
    typename AllocInner = std::allocator<T>
>
inline void load_array_to_vectors(
    const cnpy::NpyArray& array,
    Container<Container<T, AllocInner>, AllocOuter>& vec2d)
{
    if (array.shape.size() != 2)
        throw std::runtime_error("Array is not 2D");
    size_t rows = array.shape[0];
    size_t cols = array.shape[1];
    const T* data = array.data<T>();
    vec2d.resize(rows, Container<T, AllocInner>(cols));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            vec2d[i][j] = data[i * cols + j];
}

// 3D version: generic container type (default is std::vector)
template<
    typename T,
    template <typename, typename> class Container = std::vector,
    typename AllocOuter = std::allocator<Container<Container<T, std::allocator<T>>, std::allocator<Container<T, std::allocator<T>>>>>,
    typename AllocMid = std::allocator<Container<T, std::allocator<T>>>,
    typename AllocInner = std::allocator<T>
>
inline void load_array_to_vectors(
    const cnpy::NpyArray& array,
    Container<Container<Container<T, AllocInner>, AllocMid>, AllocOuter>& vec3d)
{
    if (array.shape.size() != 3)
        throw std::runtime_error("Array is not 3D");
    size_t slices = array.shape[0];
    size_t rows = array.shape[1];
    size_t cols = array.shape[2];
    const T* data = array.data<T>();
    vec3d.resize(slices, Container<Container<T, AllocInner>, AllocMid>(rows, Container<T, AllocInner>(cols)));
    for (size_t s = 0; s < slices; ++s)
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                vec3d[s][i][j] = data[s * rows * cols + i * cols + j];
}


template<typename T>
inline void load_array_to_vectors(const cnpy::NpyArray& array, std::vector<thrust::device_vector<T>>& vec2d)
{
    if (array.shape.size() != 2)
        throw std::runtime_error("Array is not 2D");
    size_t rows = array.shape[0];
    size_t cols = array.shape[1];
    const T* data = array.data<T>();
    vec2d.resize(rows);
    for (size_t i = 0; i < rows; ++i)
        vec2d[i] = thrust::device_vector<T>(data + i * cols, data + (i + 1) * cols);
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


template<typename T>
void print_one_hot_histogram_with_labels(
    const std::vector<std::vector<T>>& vec2d,
    const std::map<size_t, std::string>& labels)
{
    auto histogram = one_hot_histogram(vec2d);
    std::cout << "1-hot hist: ";
    for (const auto& kv : histogram) {
        auto it = labels.find(kv.first);
        if (it != labels.end()) {
            std::cout << it->second << ":" << kv.second << " | ";
        } else {
            std::cout << kv.first << ":" << kv.second << " | ";
        }
    }
    std::cout << std::endl;
}

/** 
 * Extract the base filename without directory path and extension.
 * For example, given "path/to/file.png", it returns "file".
 */
std::string get_base_filename(const std::string filename);


/** Plot a histogram (bar chart) and write it to an image file using OpenCV.
 *  hist: map from label index to count
 *  labels: map from label index to label string
 *  filename: output image file name (e.g., "hist.png")
 */
void plot_histogram_to_image(
    const std::map<size_t, size_t>& hist,
    const std::map<size_t, std::string>& labels,
    const std::string& filename,
    int width = 1400, int height = 480);
