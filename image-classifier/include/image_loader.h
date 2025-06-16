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
#include <tuple>


typedef uint8_t data_type_t;
typedef std::vector<std::vector<data_type_t>> Vector2D;
typedef std::vector<std::vector<std::vector<data_type_t>>> Vector3D;

/** A standard vector of device vectors on the GPU. */
typedef std::vector<thrust::device_vector<data_type_t>> DeviceVector2D;

/** A standard vector of device vectors on the GPU.  
 * 3D is the same as 2D. Pack rows and columns on 1D device vector. */
typedef std::vector<thrust::device_vector<data_type_t>> DeviceVector3D;

/**
 * @brief Labels for chest X-ray diseases.
 * These labels correspond to the indices in the one-hot encoded vectors.
 * They are used to interpret the results of the model.
 * The indices match the order in which the diseases are represented in the dataset.
 * For example, if the model predicts a vector with a '1' at index 2, it indicates the presence of "effusion".  
 */
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


/**
 * Load a 2D array from a NPY file into a vector of vectors.
 * Each row of the array becomes a separate vector.
 * @param array The NPY array to load.
 * @param vec2d The output vector of vectors.
 * @tparam T The data type of the array elements (e.g., float, int).
 * @tparam Container The container type to use for the 2D vectors (default is std::vector).
 * @tparam AllocOuter The allocator type for the outer container (default is std::allocator).
 * @tparam AllocInner The allocator type for the inner container (default is std::allocator).
 * @throws std::runtime_error if the array is not 2D.
 */
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


/**
 * Load a 3D array from a NPY file into a vector of vectors.
 * Each slice of the array becomes a separate vector, with each row being a vector.
 * @param array The NPY array to load.
 * @param vec3d The output vector of vectors.
 * @tparam T The data type of the array elements (e.g., float, int).
 * @tparam Container The container type to use for the 2D vectors (default is std::vector).
 * @tparam AllocOuter The allocator type for the outer container (default is std::allocator).
 * @tparam AllocMid The allocator type for the middle container (default is std::allocator).
 * @tparam AllocInner The allocator type for the inner container (default is std::allocator).
 * @throws std::runtime_error if the array is not 3D.
 */
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


/**
 * Load a 2D array from a NPY file into a vector of device vectors.
 * Each row of the array becomes a separate device vector.
 * @param array The NPY array to load.
 * @param vec2d The output vector of device vectors.
 * @tparam T The data type of the array elements (e.g., float, int).
 * @throws std::runtime_error if the array is not 2D.   
 */
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


/**
 * Load a 3D array from a NPY file into a vector of device vectors.
 * Each slice of the array becomes a separate device vector, flattened to 1D.
 * @param array The NPY array to load.
 * @param vec3d The output vector of device vectors.
 * @tparam T The data type of the array elements (e.g., float, int).
 * @throws std::runtime_error if the array is not 3D.
 */
template<typename T>
inline void load_array_to_vectors_3d(const cnpy::NpyArray& array, std::vector<thrust::device_vector<T>>& vec3d)
{
    if (array.shape.size() != 3)
        throw std::runtime_error("Array is not 3D");
    size_t slices = array.shape[0];
    size_t rows = array.shape[1];
    size_t cols = array.shape[2];
    const T* data = array.data<T>();
    vec3d.resize(slices);
    for (size_t s = 0; s < slices; ++s)
        vec3d[s] = thrust::device_vector<T>(data + s * rows * cols, data + (s + 1) * rows * cols);
    // Each vec3d[s] is a flattened (rows*cols) device vector for slice s
}


/**
 * Write a 2D vector to an image file using OpenCV.
 * The vector is assumed to represent a grayscale image.
 * @param array_name Base name for the output file (without extension).
 * @param vec2d 2D vector containing pixel values.
 * @param slice_numer Slice number to append to the filename.
 * @param cv_type OpenCV type for the image (default is CV_8UC1).
 * @param encoding Image format (e.g., "png", "jpg").
 */
template<typename DataType>
void write_image(const std::string& filename,
                 const std::vector<std::vector<DataType>>& vec2d,
                 const int cv_type = CV_8UC1,
                 const std::string encoding = "png") {
    // TODO: CV_8UC1 is  macro for 8-bit single channel image
    cv::Mat img(vec2d.size(), vec2d[0].size(), cv_type);
    for (size_t i = 0; i < vec2d.size(); ++i)
        for (size_t j = 0; j < vec2d[0].size(); ++j)
            img.at<DataType>(i, j) = vec2d[i][j];
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
 * Each row in vec2d is a one-hot vector; the histogram counts the index of the 'hot' (nonzero) entry. 
 * @param vec2d 2D vector where each row is a one-hot encoded vector.
 * @return A map where keys are indices of the 'hot' entries and values are their counts.
 * */ 
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


/** Print a one-hot histogram to the console.
 * Each row in vec2d is a one-hot encoded vector; the histogram counts the index of the 'hot' (nonzero) entry.
 * @param vec2d 2D vector where each row is a one-hot encoded vector.
 * @param labels Optional map from indices to label strings for better readability.
 */
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
 * @param filename Full path to the file.
 * @return Base filename without path and extension.
 */
std::string get_base_filename(const std::string filename);


/** Plot a histogram (bar chart) and write it to an image file using OpenCV.
 * @param hist Map from label index to count.
 * @param labels Map from label index to label string for better readability.
 * @param filename Output image file name (e.g., "hist.png").
 * @param width Width of the output image (default is 1400).
 * @param height Height of the output image (default is 480).
 */
void plot_histogram_to_image(
    const std::map<size_t, size_t>& hist,
    const std::map<size_t, std::string>& labels,
    const std::string& filename,
    int width = 1400, int height = 480);


/**
 * Get a vector of maps from a map of NPY arrays.
 * Each map corresponds to a single NPY array, with keys as the array names and values as the data.
 * @param all_arrays Map of NPY arrays where keys are array names and values are NpyArray objects.
 * @return A vector of maps, each containing the data from one NPY array.
 */
std::tuple<
    std::map<std::string, Vector2D>,
    std::map<std::string, Vector3D>,
    std::map<std::string, DeviceVector2D>,
    std::map<std::string, DeviceVector3D>> 
    get_vector_maps(const std::map<std::string, cnpy::NpyArray>& all_arrays);


template<typename T>
inline size_t find_hot_index(const std::vector<data_type_t>& one_hot_label) {
    auto it = std::find_if(one_hot_label.begin(), one_hot_label.end(),
                           [](T v) { return v != T(0); });
    if (it != one_hot_label.end()) {
        return std::distance(one_hot_label.begin(), it);
    }
    return 0;
}

template<typename T>
inline void plot_first_images_by_label(
    const std::map<size_t, std::string>& labels,
    const Vector3D& vec3d,
    const Vector2D& vec2d,
    const std::string& filename_prefix,
    size_t up_to_first_x = 10,
    int cv_type = CV_8UC1,
    const std::string encoding = "png")
{
    if (vec3d.empty() || vec2d.empty()) {
        std::cerr << "Error: vec3d or vec2d is empty." << std::endl;
        return;
    }
    if (vec3d.size() != vec2d.size()) {
        std::cerr << "Error: vec3d and vec2d must have the same size." << std::endl;
        return;
    }
    up_to_first_x = std::min<size_t>(up_to_first_x, vec3d.size());
    for (size_t i = 0; i < up_to_first_x; ++i) {
        auto one_hot_label = vec2d[i];
        auto hot_index = find_hot_index<T>(one_hot_label);
        std::string label = labels.at(hot_index); // Use at() to ensure it throws if not found
        std::string filename = filename_prefix + "_" + label + "_slice" + std::to_string(i) + "." + encoding;
        write_image<T>(filename, vec3d[i], cv_type, encoding);
    }
}


template<typename T>
inline void plot_one_example_per_label(
    const std::map<size_t, std::string>& labels,
    const Vector3D& vec3d,
    const Vector2D& vec2d,
    const std::string& out_filename,
    int cv_type = CV_8UC1,
    int scale = 1)
{
    if (vec3d.empty() || vec2d.empty()) {
        std::cerr << "Error: vec3d or vec2d is empty." << std::endl;
        return;
    }
    if (vec3d.size() != vec2d.size()) {
        std::cerr << "Error: vec3d and vec2d must have the same size." << std::endl;
        return;
    }

    // Find one example index for each label
    std::map<size_t, size_t> label_to_index;
    for (size_t i = 0; i < vec2d.size(); ++i) {
        size_t hot_index = find_hot_index<T>(vec2d[i]);
        if (label_to_index.count(hot_index) == 0) {
            label_to_index[hot_index] = i;
        }
    }

    size_t num_labels = label_to_index.size();
    if (num_labels == 0) {
        std::cerr << "No labels found." << std::endl;
        return;
    }

    // Assume all images are the same size
    int img_rows = vec3d[0].size();
    int img_cols = vec3d[0][0].size();

    int scaled_rows = img_rows * scale;
    int scaled_cols = img_cols * scale;

    // Create a grid image (1 row, num_labels columns)
    int grid_rows = scaled_rows;
    int grid_cols = scaled_cols * num_labels;
    cv::Mat grid_img(grid_rows, grid_cols, cv_type, cv::Scalar(0));

    size_t col = 0;
    for (const auto& kv : label_to_index) {
        size_t label_idx = kv.first;
        size_t img_idx = kv.second;
        // Convert vec3d[img_idx] to cv::Mat
        cv::Mat img(img_rows, img_cols, cv_type);
        for (int r = 0; r < img_rows; ++r) {
            for (int c = 0; c < img_cols; ++c) {
                img.at<T>(r, c) = vec3d[img_idx][r][c];
            }
        }
        // Scale the image
        cv::Mat img_scaled;
        cv::resize(img, img_scaled, cv::Size(scaled_cols, scaled_rows), 0, 0, cv::INTER_NEAREST);

        // Copy to grid
        img_scaled.copyTo(grid_img(cv::Rect(col * scaled_cols, 0, scaled_cols, scaled_rows)));
        // Put label text
        std::string label_text = labels.at(label_idx);
        cv::putText(grid_img, label_text, cv::Point(col * scaled_cols + 5, 25),
                    cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255), 2);
        ++col;
    }

    cv::imwrite(out_filename, grid_img);
}
