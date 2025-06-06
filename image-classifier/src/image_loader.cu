#include "image_loader.h"
#include <cnpy.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <vector>
#include <stdexcept>
#include <iostream>


ImageLoader::ImageLoader() 
{
}

std::map<std::string, cnpy::NpyArray> ImageLoader::read_all_npz_arrays(
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
