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


typedef uint8_t data_type_t;


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


inline auto get_vector_maps(const std::map<std::string, cnpy::NpyArray>& all_arrays) {
    std::map<std::string, std::vector<std::vector<data_type_t>>> arrays_2d;
    std::map<std::string, std::vector<std::vector<std::vector<data_type_t>>>> arrays_3d;

    std::map<std::string, std::vector<thrust::device_vector<data_type_t>>> device_arrays_2d;
    std::map<std::string, std::vector<thrust::device_vector<data_type_t>>> device_arrays_3d;

    for (const auto& pair : all_arrays) {
        const std::string& array_name = pair.first;
        const cnpy::NpyArray& array = pair.second;
        std::cout << "Array name=" << array_name << " dimensions=" << array.shape.size() << " samples=" <<
                    array.shape[0];
        if (array.shape.size() >= 2) {
            std::cout << " rows=" << array.shape[1];
        }
        if (array.shape.size() >= 3) {
            std::cout << " cols=" << array.shape[2];
        }
        if (array.shape.size() >= 4) {
            std::cout << " depth=" << array.shape[3];
        }
        std::cout << " word_size=" << array.word_size;
    
        if (array.shape.size() == 2) {
            // 2D array, likely labels
            std::vector<std::vector<data_type_t>> one_hot_labels2d;
            std::vector<thrust::device_vector<data_type_t>> dvec_one_hot2d;
            try {
                load_array_to_vectors<data_type_t>(array, one_hot_labels2d);
                arrays_2d[array_name] = one_hot_labels2d; // Store in 2D map
                load_array_to_vectors<data_type_t>(array, dvec_one_hot2d);
                device_arrays_2d[array_name] = dvec_one_hot2d; // Store in device 2D map
            } catch (const std::exception& ex) {
                std::cerr << "Error loading 2D array: " << ex.what() << std::endl;
                continue; // Skip to next array
            }
            std::cout << " Loaded 2D labels with " << one_hot_labels2d.size() << " rows.";
            auto label_histogram = one_hot_histogram<data_type_t>(one_hot_labels2d);
            std::cout << "\n1-hot hist: ";
            for (const auto& kv : label_histogram) {
                std::cout << kv.first << ":" << kv.second << " | ";
            }
            std::cout << std::endl;
            print_one_hot_histogram_with_labels<data_type_t>(one_hot_labels2d, CHEST_LABELS);
            plot_histogram_to_image(label_histogram, CHEST_LABELS, array_name + "_hist.png");
        } else if (array.shape.size() == 3) {
            // 3D array, likely images
            std::vector<std::vector<std::vector<data_type_t>>> list_image;
            std::vector<thrust::device_vector<data_type_t>> dvec_image;
            std::cout << " Loading 3D image data.";
            try {
                load_array_to_vectors<data_type_t>(array, list_image);
                arrays_3d[array_name] = list_image; // Store in 3D map
                load_array_to_vectors_3d<data_type_t>(array, dvec_image);
                device_arrays_3d[array_name] = dvec_image; // Store in device 3D map
            } catch (const std::exception& ex) {
                std::cerr << "Error loading 3D array: " << ex.what() << std::endl;
                continue; // Skip to next array
            }
            std::cout << " Loaded 3D image with " << list_image.size() << " slices.";
            std::cout << " First 3D value=" << static_cast<int>(list_image[0][0][0]);
            const int slice_number = 0;
            write_image<data_type_t>(array_name, list_image[slice_number], slice_number, CV_8UC1, "png");
        } else {
            std::cerr << "Unsupported array shape size: " << array.shape.size() << std::endl;
            continue; // Skip unsupported shapes
        }
        std::cout << std::endl;
    }
    return std::make_tuple(arrays_2d, arrays_3d, device_arrays_2d, device_arrays_3d);
}