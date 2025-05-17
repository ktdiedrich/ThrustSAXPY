cmake_minimum_required(VERSION 3.10)
project(ImageClassifier LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

include_directories(include)
include_directories(src)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test_gtest tests/test_gtest.cpp)
target_link_libraries(test_gtest gtest_main)

set(SOURCES
    src/main.cu
    src/classifier.cu
    src/image_loader.cu
)

add_executable(image_classifier ${SOURCES})
target_compile_options(image_classifier PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
target_link_libraries(image_classifier PRIVATE ${OpenCV_LIBS})
