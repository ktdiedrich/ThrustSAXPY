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
