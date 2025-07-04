cmake_minimum_required(VERSION 3.10)
project(ThrustSAXPY LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)

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

include_directories(include)

add_executable(detect_compute_capability src/detect_compute_capability.cpp)
# CMAKE_CUDA_ARCHITECTURES will be autodetected by cmake
add_executable(thrustNN src/thrustNN.cu)
add_executable(thrustSAXPY src/thrustSAXPY.cu)

target_compile_options(thrustNN PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
target_compile_options(thrustSAXPY PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

target_link_libraries(thrustSAXPY ${CUDA_CUDA_LIBRARIES})
target_link_libraries(thrustNN ${CUDA_CUDA_LIBRARIES})

# Add CUDA runtime library and custom include/library paths for detect_compute_capability
target_link_libraries(detect_compute_capability PRIVATE cudart)
target_include_directories(detect_compute_capability PRIVATE /usr/include)
target_link_directories(detect_compute_capability PRIVATE /lib/x86_64-linux-gnu)

# Build subdirectories
add_subdirectory(image-classifier)
