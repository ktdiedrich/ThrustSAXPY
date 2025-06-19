cmake_minimum_required(VERSION 3.10)
project(ImageClassifier LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_path(CUDNN_INCLUDE_DIR cudnn.h PATHS ${CUDAToolkit_INCLUDE_DIRS} /usr/include /usr/local/cuda/include)
find_library(CUDNN_LIBRARY cudnn PATHS ${CUDAToolkit_LIBRARY_DIR} /usr/lib /usr/local/cuda/lib64)

if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN not found!")
endif()


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)

FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
  GIT_TAG master
)
FetchContent_MakeAvailable(cnpy)

include_directories(include)
include_directories(src)
include_directories(${CUDNN_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable(test_gtest tests/test_gtest.cpp)
target_link_libraries(test_gtest gtest_main)

include_directories(${CUDNN_INCLUDE_DIR})
add_executable(test_cudnn_gtest tests/test_cudnn_gtest.cu)
target_include_directories(test_cudnn_gtest PRIVATE ${CUDNN_INCLUDE_DIR})
target_compile_options(test_cudnn_gtest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
target_link_libraries(test_cudnn_gtest PRIVATE ${CUDNN_LIBRARY} gtest_main pthread ${CUDA_LIBRARIES})

gtest_discover_tests(test_gtest)

set(SOURCES
    src/main.cu
    src/classifier.cu
    src/image_loader.cu
)

add_executable(image_classifier ${SOURCES})
target_include_directories(image_classifier PRIVATE ${cnpy_SOURCE_DIR} ${CUDNN_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_compile_options(image_classifier PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
target_link_libraries(image_classifier PRIVATE ${OpenCV_LIBS} cnpy z ${CUDNN_LIBRARY} ${CUDA_LIBRARIES} pthread)
