#pragma once

#include <vector>
#include <thrust/device_vector.h>

class ImageClassifier {
public:
    ImageClassifier(int input_size);

    // Train with images (flattened, normalized) and labels (0 or 1)
    void train(const std::vector<thrust::device_vector<float>>& images, const std::vector<int>& labels, int epochs = 10, float lr = 0.01f);

    // Predict labels for images
    std::vector<int> predict(const std::vector<thrust::device_vector<float>>& images);

private:
    thrust::device_vector<float> weights_;
    float bias_;
};
