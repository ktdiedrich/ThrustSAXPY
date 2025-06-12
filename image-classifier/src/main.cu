#include "image_loader.h"
#include "classifier.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cnpy.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_file NPZ>" << std::endl;
        return 1;
    }
    std::map<std::string, cnpy::NpyArray> all_arrays = read_all_npz_arrays(argv[1]);
    
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
            std::vector<std::vector<uint8_t>> one_hot_labels2d;
            std::vector<thrust::device_vector<uint8_t>> dvec_one_hot2d;
            try {
                load_array_to_vectors<uint8_t>(array, one_hot_labels2d);
                load_array_to_vectors<uint8_t>(array, dvec_one_hot2d);
            } catch (const std::exception& ex) {
                std::cerr << "Error loading 2D array: " << ex.what() << std::endl;
                continue; // Skip to next array
            }
            std::cout << " Loaded 2D labels with " << one_hot_labels2d.size() << " rows.";
            auto label_histogram = one_hot_histogram<uint8_t>(one_hot_labels2d);
            std::cout << "\n1-hot hist: ";
            for (const auto& kv : label_histogram) {
                std::cout << kv.first << ":" << kv.second << " | ";
            }
            std::cout << std::endl;
            print_one_hot_histogram_with_labels<uint8_t>(one_hot_labels2d, CHEST_LABELS);
            plot_histogram_to_image(label_histogram, CHEST_LABELS, array_name + "_hist.png");
        } else if (array.shape.size() == 3) {
            // 3D array, likely images
            std::vector<std::vector<std::vector<uint8_t>>> list_image;
            std::cout << " Loading 3D image data.";
            try {
                load_array_to_vectors<uint8_t>(array, list_image);
            } catch (const std::exception& ex) {
                std::cerr << "Error loading 3D array: " << ex.what() << std::endl;
                continue; // Skip to next array
            }
            std::cout << " Loaded 3D image with " << list_image.size() << " slices.";
            std::cout << " First 3D value=" << static_cast<int>(list_image[0][0][0]);
            const int slice_number = 0;
            write_image<uint8_t>(array_name, list_image[slice_number], slice_number, CV_8UC1, "png");
        } else {
            std::cerr << "Unsupported array shape size: " << array.shape.size() << std::endl;
            continue; // Skip unsupported shapes
        }
        std::cout << std::endl;
    }
    return 0;
}
