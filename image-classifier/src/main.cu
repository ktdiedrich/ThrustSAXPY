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

    std::map<std::string, std::vector<std::vector<data_type_t>>> arrays_2d;
    std::map<std::string, std::vector<std::vector<std::vector<data_type_t>>>> arrays_3d;
    std::map<std::string, std::vector<thrust::device_vector<data_type_t>>> device_arrays_2d;
    std::map<std::string, std::vector<thrust::device_vector<data_type_t>>> device_arrays_3d;
    std::tie(arrays_2d, arrays_3d, device_arrays_2d, device_arrays_3d) = get_vector_maps(all_arrays);
    
    return 0;
}
