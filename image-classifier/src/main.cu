#include "image_loader.h"
#include "classifier.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cnpy.h>


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_file NPZ>" << std::endl;
        return 1;
    }
    ImageLoader loader;
    std::map<std::string, cnpy::NpyArray> all_arrays = loader.read_all_npz_arrays(argv[1]);
    
    for (const auto& pair : all_arrays) {
        const std::string& array_name = pair.first;
        const cnpy::NpyArray& array = pair.second;
        std::cout << "Array name=" << array_name << " dimensions=" << array.shape.size() << " n samples=" <<
                    array.shape[0];
        if (array.shape.size() >= 2) {
            std::cout << " rows=" << array.shape[1];
        }
        if (array.shape.size() >= 3) {
            std::cout << " cols=" << array.shape[2];
        }
        if (array.shape.size() >= 4) {
            std::cout << " channels=" << array.shape[3];
        }
        std::cout << std::endl;
    }    
    return 0;
}
