#include "image_loader.h"
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <vector>
#include <stdexcept>

ImageLoader::ImageLoader(const std::string& directory) : imageDirectory(directory) {}

std::vector<thrust::device_vector<float>> ImageLoader::load_images(const std::vector<std::string>& filenames, int width, int height) {
    std::vector<thrust::device_vector<float>> images;
    for (const auto& filename : filenames) {
        cv::Mat img = cv::imread(imageDirectory + "/" + filename, cv::IMREAD_COLOR);
        if (img.empty()) {
            throw std::runtime_error("Could not open or find the image: " + filename);
        }
        cv::Mat img_resized;
        cv::resize(img, img_resized, cv::Size(width, height));
        img_resized.convertTo(img_resized, CV_32F, 1.0 / 255);

        // Flatten to 1D float array
        std::vector<float> img_flat;
        img_flat.assign((float*)img_resized.datastart, (float*)img_resized.dataend);

        thrust::device_vector<float> img_vec(img_flat.begin(), img_flat.end());
        images.push_back(std::move(img_vec));
    }
    return images;
}
