#include "image_loader.h"
#include "classifier.h"
#include <iostream>
#include <vector>
#include <iomanip>


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_file NPZ>" << std::endl;
        return 1;
    }
    ImageLoader loader;
    auto all_arrays = loader.read_all_npz_arrays(argv[1]);
    
    for (const auto& pair : all_arrays) {
        const std::string& array_name = pair.first;
        const auto& vectors = pair.second;
        std::cout << "Array: " << array_name << ", rows: " << vectors.size() << std::endl;
        if (!vectors.empty()) {
            std::cout << "  First row (first 10 values): ";
            std::vector<float> host_row(vectors[0].size());
            thrust::copy(vectors[0].begin(), vectors[0].end(), host_row.begin());
            for (size_t i = 0; i < std::min<size_t>(10, host_row.size()); ++i) {
                std::cout << std::setprecision(4) << host_row[i] << " ";
            }
            std::cout << std::endl;
        }
    }    
    return 0;
}
