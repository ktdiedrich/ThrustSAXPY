cmake_minimum_required(VERSION 3.10)
project(ImageClassifier LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

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
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test_gtest tests/test_gtest.cpp)
target_link_libraries(test_gtest gtest_main)

gtest_discover_tests(test_gtest)

set(SOURCES
    src/main.cu
    src/classifier.cu
    src/image_loader.cu
)

add_executable(image_classifier ${SOURCES})
target_include_directories(image_classifier PRIVATE ${cnpy_SOURCE_DIR})
target_compile_options(image_classifier PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
target_link_libraries(image_classifier PRIVATE ${OpenCV_LIBS} cnpy z)
