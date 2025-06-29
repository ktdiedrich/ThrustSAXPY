#pragma once

#include <vector>
#include <thrust/device_vector.h>
#include <cudnn.h>
#include <iostream>
#include "image_loader.h"


void train_validate_classifier(const DeviceVector3D& train_images,
                                     const DeviceVector2D& train_labels,
                                     const DeviceVector3D& val_images,
                                     const DeviceVector2D& val_labels,
                                     const std::string& model_path,
                                     const std::string& output_path, 
                                    const float learning_rate,
                                    const int epochs,
                                    const int batch_size,
                                    const float momentum);


// Helper: check cuDNN status
#define checkCUDNN(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    }

void train_validate_classifier_cudnn(
    const thrust::device_vector<float>& train_images,   // [N * C * H * W]
    const thrust::device_vector<float>& train_labels,   // [N * num_classes]
    const thrust::device_vector<float>& val_images,
    const thrust::device_vector<float>& val_labels,
    int N, int C, int H, int W, int num_classes,
    int epochs, int batch_size);
