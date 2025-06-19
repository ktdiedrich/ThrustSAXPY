#include "image_loader.h"
#include "classifier.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cnpy.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>



int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_file NPZ>" << std::endl;
        return 1;
    }
    std::map<std::string, cnpy::NpyArray> all_arrays = read_all_npz_arrays(argv[1]);

    std::map<std::string, Vector2D> one_hot_labels_2d;
    std::map<std::string, Vector3D> images_3d;
    std::map<std::string, DeviceVector2D> device_one_hot_labels_2d;
    std::map<std::string, DeviceVector3D> device_images_3d;
    std::tie(one_hot_labels_2d, images_3d, device_one_hot_labels_2d, device_images_3d) = get_vector_maps(all_arrays);
    
    plot_one_example_per_label<data_type_t>(
        CHEST_LABELS,
        images_3d["train_images"],
        one_hot_labels_2d["train_labels"],
        "class_examples_train_images.png",
        CV_8UC1,
        4
    );

    std::string model_path = "trained_model"; // Placeholder for model path
    std::string output_path = "training_output"; // Placeholder for output path
    train_validate_classifier(
        device_images_3d["train_images"],
        device_one_hot_labels_2d["train_labels"],
        device_images_3d["val_images"],
        device_one_hot_labels_2d["val_labels"],
        model_path,
        output_path,
        0.001f, // learning rate
        10,     // epochs
        32,     // batch size
        0.9f    // momentum
    );
    return 0;
}
