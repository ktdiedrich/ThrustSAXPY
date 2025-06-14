#include "image_loader.h"
#include <cnpy.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <vector>
#include <stdexcept>
#include <iostream>


std::map<std::string, cnpy::NpyArray> read_all_npz_arrays(
    const std::string& data_file) {
    std::map<std::string, cnpy::NpyArray> all_arrays;
    cnpy::npz_t npz = cnpy::npz_load(data_file);
    
    for (const auto& kv : npz) {
        const std::string& array_name = kv.first;
        cnpy::NpyArray arr = kv.second;
        all_arrays[array_name] = arr;
    }
    return all_arrays;
}

std::string get_base_filename(const std::string filename) {
    std::string base_title = filename;
    // Remove directory path
    size_t last_slash = filename.find_last_of("/\\");
    if (last_slash != std::string::npos)
        base_title = base_title.substr(last_slash + 1);
    // Remove extension (e.g., ".png")
    size_t last_dot = base_title.find_last_of('.');
    if (last_dot != std::string::npos)
        base_title = base_title.substr(0, last_dot);
    return base_title;
}


void plot_histogram_to_image(
    const std::map<size_t, size_t>& hist,
    const std::map<size_t, std::string>& labels,
    const std::string& filename,
    int width, int height)
{
    // Find max count for scaling
    size_t max_count = 1;
    for (const auto& kv : hist) max_count = std::max(max_count, kv.second);

    int margin = 40;
    int bar_width = (width - 2 * margin) / std::max<size_t>(1, hist.size());
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(255,255,255));

    // Draw title (base filename without extension)
    std::string base_title = get_base_filename(filename);
    cv::putText(img, base_title, cv::Point(margin, margin - 10), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(10,10,10), 2);

    int i = 0;
    for (const auto& kv : hist) {
        int x1 = margin + i * bar_width;
        int x2 = x1 + bar_width - 4;
        int y2 = height - margin;
        int y1 = y2 - static_cast<int>((double(kv.second) / max_count) * (height - 2 * margin));
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(100, 100, 240), cv::FILLED);

        // Draw label
        std::string label = std::to_string(kv.first);
        auto it = labels.find(kv.first);
        if (it != labels.end()) label = it->second;
        cv::putText(img, label, cv::Point(x1, height - margin + 18), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);

        // Draw count
        cv::putText(img, std::to_string(kv.second), cv::Point(x1, y1 - 6), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);

        ++i;
    }
    // Draw axes
    cv::line(img, cv::Point(margin, margin), cv::Point(margin, height - margin), cv::Scalar(0,0,0), 1);
    cv::line(img, cv::Point(margin, height - margin), cv::Point(width - margin, height - margin), cv::Scalar(0,0,0), 1);

    cv::imwrite(filename, img);
    std::cout << "Histogram image written to: " << filename << std::endl;
}


std::tuple<
    std::map<std::string, Vector2D>,
    std::map<std::string, Vector3D>,
    std::map<std::string, DeviceVector2D>,
    std::map<std::string, DeviceVector3D>>
get_vector_maps(const std::map<std::string, cnpy::NpyArray>& all_arrays) {
    std::map<std::string, Vector2D> arrays_2d;
    std::map<std::string, Vector3D> arrays_3d;

    std::map<std::string, DeviceVector2D> device_arrays_2d;
    std::map<std::string, DeviceVector3D> device_arrays_3d;

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
            Vector2D one_hot_labels2d;
            DeviceVector2D dvec_one_hot2d;
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
            Vector3D list_image;
            DeviceVector3D dvec_image;
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
            std::string encoding = "png";
            std::string filename = array_name + "_slice_" + std::to_string(slice_number) + "." + encoding;
            write_image<data_type_t>(filename, list_image[slice_number], CV_8UC1, encoding);
        } else {
            std::cerr << "Unsupported array shape size: " << array.shape.size() << std::endl;
            continue; // Skip unsupported shapes
        }
        std::cout << std::endl;
    }
    return std::make_tuple(arrays_2d, arrays_3d, device_arrays_2d, device_arrays_3d);
}
