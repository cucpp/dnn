################################################################################
#    Examples of cucpp/dnn API usage.
################################################################################

# Append cuDNN++ include directory to include path
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../include")
# Append CUDA & CUDNN include directories to include path
include_directories(${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE})
# Link against CUDA & CUDNN runtime libraries
link_libraries(${CUDA_LIBRARIES} ${CUDNN_LIBRARY})

# Examples:

add_executable(Convolution "Convolution.cpp")
