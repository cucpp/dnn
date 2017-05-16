################################################################################
#    cucpp/dnn - Single-header C++ wrapper for NVIDIA cuDNN library. 
################################################################################

cmake_minimum_required (VERSION 3.2)

project(cuDNN++ CXX)

# Use custom CMake modeuls
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Use C++11 Standard
set (CMAKE_CXX_STANDARD 11)

################################################################################
#    Find Required Packages
################################################################################

find_package(CUDA REQUIRED)
find_package(CUDNN REQUIRED)

################################################################################
#    Build cucpp/dnn examples.
################################################################################

add_subdirectory(examples)
