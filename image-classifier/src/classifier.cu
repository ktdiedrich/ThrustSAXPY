#include "classifier.h"
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <cmath>

namespace {
struct sigmoid_functor {
    __host__ __device__
    float operator()(float x) const {
        return 1.0f / (1.0f + expf(-x));
    }
};
}

ImageClassifier::ImageClassifier(int input_size)
    : weights_(input_size, 0.01f), bias_(0.0f) {}

void ImageClassifier::train(const std::vector<thrust::device_vector<float>>& images, const std::vector<int>& labels, int epochs, float lr) {
    int n = images.size();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < n; ++i) {
            // Linear combination
            float z = thrust::inner_product(images[i].begin(), images[i].end(), weights_.begin(), bias_);
            // Sigmoid
            float pred = 1.0f / (1.0f + std::exp(-z));
            // Gradient
            float error = pred - labels[i];
            // Update weights and bias
            thrust::transform(
                weights_.begin(), weights_.end(), images[i].begin(), weights_.begin(),
                [lr, error] __host__ __device__ (float w, float x) { return w - lr * error * x; }
            );
            bias_ -= lr * error;
        }
    }
}

std::vector<int> ImageClassifier::predict(const std::vector<thrust::device_vector<float>>& images) {
    std::vector<int> results;
    for (const auto& img : images) {
        float z = thrust::inner_product(img.begin(), img.end(), weights_.begin(), bias_);
        float pred = 1.0f / (1.0f + std::exp(-z));
        results.push_back(pred > 0.5f ? 1 : 0);
    }
    return results;
}
