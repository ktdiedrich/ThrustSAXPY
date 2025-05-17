cmake_minimum_required(VERSION 3.10)
project(ImageClassifier LANGUAGES CXX CUDA)

set(CMAKE_C_COMPILER /usr/bin/gcc-12)
set(CMAKE_CXX_COMPILER /usr/bin/g++-12)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# Download and unpack GoogleTest at configure time
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/refs/heads/main.zip
# )
# # For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)

# enable_testing()


include_directories(include)
include_directories(src)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCES "src/*.cu")
list(FILTER SOURCES EXCLUDE REGEX "${CMAKE_BINARY_DIR}/")

add_executable(image_classifier ${SOURCES})

target_compile_options(image_classifier PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
target_link_libraries(image_classifier PRIVATE ${OpenCV_LIBS})

# add_executable(test_gtest tests/test_gtest.cpp)
# target_link_libraries(test_gtest gtest_main)

# include(GoogleTest)
# gtest_discover_tests(test_gtest)
