#pragma once

#include <string>
#include <vector>
#include <thrust/device_vector.h>
#include <map>

/** 
 * @brief ImageLoader class for loading images from NPZ files.
*/
class ImageLoader {
public:
    /** @brief Constructor for ImageLoader.
     */
    explicit ImageLoader();

    /** @brief Load data from compressed files.
     * @param data_file Path to the NPZ file.
     * @return A vector of device vectors containing the loaded data.
     */
    std::map<std::string, std::vector<thrust::device_vector<float>>> read_all_npz_arrays(const std::string& data_file);
};
