#include "image_loader.h"
#include "classifier.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_directory> <img1> <img2> ..." << std::endl;
        return 1;
    }
    ImageLoader loader(argv[1]);
    std::vector<std::string> filenames;
    for (int i = 2; i < argc; ++i) filenames.push_back(argv[i]);
    auto images = loader.load_images(filenames);

    // Dummy labels for demonstration
    std::vector<int> labels(images.size(), 0);
    if (!labels.empty()) labels[0] = 1;

    ImageClassifier clf(images[0].size());
    clf.train(images, labels, 10, 0.01f);

    auto preds = clf.predict(images);
    for (size_t i = 0; i < preds.size(); ++i)
        std::cout << filenames[i] << ": " << preds[i] << std::endl;
    return 0;
}