# Image Classifier

This project implements an image classifier using Thrust for parallel processing on NVIDIA GPUs. The classifier is designed to load images, preprocess them, and perform classification tasks efficiently.

## Project Structure

```
image-classifier
├── src
│   ├── main.cu               # Entry point of the application
|--- include
├── CMakeLists.txt             # CMake configuration file
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd ThrustSAXPY/image-classifier
   ```

2. **Install dependencies:**
   Ensure you have CMake and a compatible NVIDIA CUDA toolkit installed.

3. **Build the project:**
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage

To run the image classifier, execute the following command from the build directory:

```
./image-classifier
```

## Examples

- Training the classifier with a dataset:
  - Place your training images in the designated folder and modify the paths in `main.cu` accordingly.

- Classifying new images:
  - Load new images using the `ImageLoader` class and classify them using the `ImageClassifier`.

## image-classifer Python environment
```
 conda env create -f environment.yml
conda activate image-classifier
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.