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
    
        std::vector<std::vector<uint8_t>> vec2d;
        std::vector<std::vector<std::vector<uint8_t>>> vec3d;
        try {
            load_array_to_vectors<uint8_t>(array, vec2d, vec3d);
            // Example: print first value if available
            if (!vec2d.empty() && !vec2d[0].empty()) 
                std::cout << " First 2D value=" << static_cast<int>(vec2d[0][0]);
            if (!vec3d.empty() && !vec3d[0].empty() && !vec3d[0][0].empty()) {
                std::cout << " First 3D value=" << static_cast<int>(vec3d[0][0][0]);
                const int slice_number = 0;
                write_image<uint8_t>(array_name, vec3d[slice_number], slice_number, CV_8UC1, "png");
            }
        } catch (const std::exception& ex) {
            std::cerr << "Error loading array: " << ex.what() << std::endl;
        }
        std::cout << std::endl;
    }   
    
    return 0;
}
