#include "image_loader.h"
#include "classifier.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cnpy.h>
#include <opencv2/opencv.hpp>


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
    
        std::vector<std::vector<uint8_t>> one_hot_labels2d;
        std::vector<std::vector<std::vector<uint8_t>>> list_image;
        try {
            load_array_to_vectors<uint8_t>(array, one_hot_labels2d, list_image);
            // Example: print first value if available
            if (!one_hot_labels2d.empty() && !one_hot_labels2d[0].empty()) { 
                std::cout << " Label vec2d[0]=";
                for (size_t j = 0; j < one_hot_labels2d[0].size(); ++j) {
                    std::cout << static_cast<int>(one_hot_labels2d[0][j]) << " ";
                }
                auto label_histogram = one_hot_histogram<uint8_t>(one_hot_labels2d);
                std::cout << "\n1-hot hist: ";
                for (const auto& kv : label_histogram) {
                    std::cout << kv.first << ":" << kv.second << " | ";
                }
                std::cout << std::endl;
                print_one_hot_histogram_with_labels<uint8_t>(one_hot_labels2d, CHEST_LABELS);
                plot_histogram_to_image(label_histogram, CHEST_LABELS, array_name + "_hist.png");
            } else {
                std::cout << " No 2D data available.";  
            }
            if (!list_image.empty() && !list_image[0].empty() && !list_image[0][0].empty()) {
                std::cout << " First 3D value=" << static_cast<int>(list_image[0][0][0]);
                const int slice_number = 0;
                write_image<uint8_t>(array_name, list_image[slice_number], slice_number, CV_8UC1, "png");
            }
        } catch (const std::exception& ex) {
            std::cerr << "Error loading array: " << ex.what() << std::endl;
        }
        std::cout << std::endl;
    }   
    
    return 0;
}
