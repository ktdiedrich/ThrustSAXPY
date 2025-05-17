#pragma once

#include <string>
#include <vector>
#include <thrust/device_vector.h>

class ImageLoader {
public:
    explicit ImageLoader(const std::string& directory);

    // Loads all images in the directory, resizes and normalizes them
    std::vector<thrust::device_vector<float>> load_images(const std::vector<std::string>& filenames, int width = 224, int height = 224);

    std::string imageDirectory;
};
