/*!  
    @file cudnn.hpp

    NVIDIA CuDNN C++ single header wrapper-library.

Licensed under the MIT License <http://opensource.org/licenses/MIT>.
Copyright (c) 2017 Konstantyn Komarov.

Permission is hereby  granted, free of charge, to any  person obtaining a copy
of this software and associated  documentation files (the "Software"), to deal
in the Software  without restriction, including without  limitation the rights
to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef INCLUDE_CUDNN_HPP_
#define INCLUDE_CUDNN_HPP_

//////////////////////////////////////////////////////////////////////////////
// NVIDIA headers
//////////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cudnn.h>

//////////////////////////////////////////////////////////////////////////////
// STL headers
//////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

namespace CuDNN {

typedef cudnnActivationMode_t ActivationMode;
typedef cudnnBatchNormMode_t BatchNormMode;
typedef cudnnConvolutionBwdDataAlgo_t ConvolutionBwdDataAlgo;
typedef cudnnConvolutionBwdDataAlgoPerf_t ConvolutionBwdDataAlgoPerf;
typedef cudnnConvolutionBwdDataPreference_t ConvolutionBwdDataPreference;
typedef cudnnConvolutionBwdFilterAlgo_t ConvolutionBwdFilterAlgo;
typedef cudnnConvolutionBwdFilterAlgoPerf_t ConvolutionBwdFilterAlgoPerf;
typedef cudnnConvolutionBwdFilterPreference_t ConvolutionBwdFilterPreference;
typedef cudnnConvolutionFwdAlgo_t ConvolutionFwdAlgo;
typedef cudnnConvolutionFwdAlgoPerf_t ConvolutionFwdAlgoPerf;
typedef cudnnConvolutionFwdPreference_t ConvolutionFwdPreference;
typedef cudnnConvolutionMode_t ConvolutionMode;
typedef cudnnStatus_t Status;
typedef cudnnDataType_t DataType;
typedef cudnnDirectionMode_t DirectionMode;
typedef cudnnDivNormMode_t DivNormMode;
typedef cudnnLRNMode_t LRNMode;
typedef cudnnNanPropagation_t NanPropagation;
typedef cudnnOpTensorOp_t OpTensorOp;
typedef cudnnPoolingMode_t PoolingMode;
typedef cudnnRNNInputMode_t RNNInputMode;
typedef cudnnRNNMode_t RNNMode;
typedef cudnnSamplerType_t SamplerType;
typedef cudnnSoftmaxAlgorithm_t SoftmaxAlgorithm;
typedef cudnnSoftmaxMode_t SoftmaxMode;
typedef cudnnStatus_t Status;
typedef cudnnTensorFormat_t TensorFormat;

//////////////////////////////////////////////////////////////////////////////
// CuDNN Exception
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN exception class that is used as a C++ alternative for 
 *             C-style status codes.
 * 
 */
class Exception {
    /**
     * CuDNN operation status code.
     */
    Status mStatus;

 public:
    /**
     * @brief      Constructs the exception object with a provided status code.
     *
     * @param[in]  status  CuDNN operation status.
     */
    explicit Exception(Status status) : mStatus(status) {}
    /**
     * @brief      Returns a string description of the exception.
     *
     * @return     String desctription of the exception.
     */
    const char* what() const noexcept {
        switch (mStatus) {
            case CUDNN_STATUS_SUCCESS:
                return "CuDNN::Exception: CUDNN_STATUS_SUCCESS";
            case CUDNN_STATUS_NOT_INITIALIZED:
                return "CuDNN::Exception: CUDNN_STATUS_NOT_INITIALIZED";
            case CUDNN_STATUS_ALLOC_FAILED:
                return "CuDNN::Exception: CUDNN_STATUS_ALLOC_FAILED";
            case CUDNN_STATUS_BAD_PARAM:
                return "CuDNN::Exception: CUDNN_STATUS_BAD_PARAM";
            case CUDNN_STATUS_ARCH_MISMATCH:
                return "CuDNN::Exception: CUDNN_STATUS_ARCH_MISMATCH";
            case CUDNN_STATUS_MAPPING_ERROR:
                return "CuDNN::Exception: CUDNN_STATUS_MAPPING_ERROR";
            case CUDNN_STATUS_EXECUTION_FAILED:
                return "CuDNN::Exception: CUDNN_STATUS_EXECUTION_FAILED";
            case CUDNN_STATUS_INTERNAL_ERROR:
                return "CuDNN::Exception: CUDNN_STATUS_INTERNAL_ERROR";
            case CUDNN_STATUS_NOT_SUPPORTED:
                return "CuDNN::Exception: CUDNN_STATUS_NOT_SUPPORTED";
            case CUDNN_STATUS_LICENSE_ERROR:
                return "CuDNN::Exception: CUDNN_STATUS_LICENSE_ERROR";
            default:
                return "CuDNN::Exception: CUDNN_STATUS_UNKNOWN";
        }
    };

    /**
     * @brief      Returns a CuDNN operation status that caused this exception.
     *
     * @return     CuDNN operation status.
     */
    Status getStatus() const {
        return mStatus;
    }
};

/**
 * @brief      Helper function for checking CuDNN operation status.
 * @throw      Exception Throws if status is not equal to CUDNN_STATUS_SUCCESS.
 *
 * @param[in]  status  CuDNN operation status.
 */
void checkStatus(Status status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        Exception e = Exception(status);
        throw e;
    }
}

//////////////////////////////////////////////////////////////////////////////
// CuDNN detail RAII
//////////////////////////////////////////////////////////////////////////////

/**
 * @internal
 * @namespace  Namespace that contains CuDNN implementation details.
 */
namespace detail {

/**
 * @internal
 * @brief      Utility class that helps CuDNN objects conform to 
 *             <a href="http://en.cppreference.com/w/cpp/language/raii">RAII</a> 
 *             principle.
 *
 * @tparam     T     cuDNN C-style structure.
 * @tparam     C     cuDNN constructor function. 
 * @tparam     D     cuDNN destructor function.
 * @endinternal
 */
template <typename T, Status (*C)(T*), Status (*D)(T)>
class RAII {
    /**
     * @brief      Resource object that manages the lifetime of cuDNN structure.
     */
    struct Resource {
        /**
         * C-style object.
         */
        T object;

        /**
         * @brief      Constructs the resource object.
         */
        Resource() {
            checkStatus(C(&object));
        }

        /**
         * @brief      Destroys the resource object.
         */
        ~Resource() {
            checkStatus(D(object));
        }
    };

    /**
     * Pointer to the resource object.
     */
    std::shared_ptr<Resource> mResource;

 protected:
    /**
     * @brief      Constructs the RAII object.
     */
    RAII() : mResource(std::make_shared<Resource>()) {}

 public:
    /**
     * @brief      Returns the underlying cuDNN structure.
     *
     * @return     C-style cuDNN structure.
     */
    T get() const {
        return mResource->object;
    }

    /**
     * @brief      Defines conversion betwen RAII-compliant objects and
     *             C-style cuDNN structures.  
     */
    operator T() const {
      return mResource->object;
    }
};

}   // namespace detail

//////////////////////////////////////////////////////////////////////////////
// CuDNN Handle
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN library handle.
 */
class Handle :
    public detail::RAII<cudnnHandle_t,
                        cudnnCreate,
                        cudnnDestroy> {};

//////////////////////////////////////////////////////////////////////////////
// CuDNN detail Buffer
//////////////////////////////////////////////////////////////////////////////

namespace detail {

/**
 * @brief      CuDNN GPU-data buffer class.
 *
 * @tparam     T     The underlying data type (float or double) 
 */
template <typename T>
class Buffer {
    /**
     * @brief      Internally used CUDA buffer.
     */
    struct CudaBuffer {
        T* mData;
        size_t mSize;
        /**
         * @brief      Constructs the CUDA buffer object.
         *
         * @param[in]  size  Buffer size in number of elements.
         */
        explicit CudaBuffer(size_t size) : mSize(size) {
            cudaMalloc(&mData, size * sizeof(T));
        }

        /**
         * @brief      Destroys the CUDA buffer object.
         */
        ~CudaBuffer() {
            cudaFree(mData);
        }
    };

    /**
     * Pointer to the CUDA buffer object.
     */
    std::shared_ptr<CudaBuffer> mBuffer;

 public:
    /**
     * @brief      Constructs the buffer object.
     *
     * @param[in]  size  Size of the buffer
     */
    explicit Buffer (size_t size) : mBuffer(new CudaBuffer(size)) {
        static_assert(  // make sure correct data type is provided
		      std::is_same<T, float>::value ||
		      std::is_same<T, double>::value,
		      "Invalid CuDNN::detail::Buffer<T> data-type");
    }

    /**
     * @brief      Defines conversion between Buffer objects and 
     *             C-style pointers to the underlying data. 
     */
    operator T*() const {
        return mBuffer->mData;
    }

    /**
     * @brief      Gets the buffer size.
     * @note       Size is defined as number of elements
     *             (e.g. not as number of bytes in memory).
     *
     * @return     Size of the buffer in number of elements.
     */
    size_t getSize() const {
        return mBuffer->mSize;
    }
};

}   // namespace detail

//////////////////////////////////////////////////////////////////////////////
// CuDNN detail dataType
//////////////////////////////////////////////////////////////////////////////

namespace detail {
/**
 * @internal
 * @brief      Trait class that maps template data into corresponding cuDNN 
 *             enums at compile time.
 *
 * @tparam     T     Type to convert (float or double)
 * @endinternal
 */
template <typename T> struct dataType {};

/**
 * @internal
 * @brief      Template specialization for float.
 */
template<>
struct dataType<float> {
    static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
};

/**
 * @internal
 * @brief      Template specialization for double.
 */
template <>
struct dataType<double> {
    static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
};
}   // namespace detail

//////////////////////////////////////////////////////////////////////////////
// CuDNN Activation Descriptor
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN activation descriptor class.
 */
class ActivationDescriptor :
    public detail::RAII<cudnnActivationDescriptor_t,
                        cudnnCreateActivationDescriptor,
                        cudnnDestroyActivationDescriptor> {};

//////////////////////////////////////////////////////////////////////////////
// CuDNN Convolution Descriptor
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN convolution descriptor class.
 */
template<typename T>
class ConvolutionDescriptor :
    public detail::RAII<cudnnConvolutionDescriptor_t,
                        cudnnCreateConvolutionDescriptor,
                        cudnnDestroyConvolutionDescriptor> {
    ConvolutionDescriptor() {}

 public:
    /**
     * @brief      Creates a new descriptor object for CuDNN convolution.
     *
     * @param[in]  padH      Padding height.
     * @param[in]  padW      Padding width.
     * @param[in]  strideH   Stride height.
     * @param[in]  strideW   Stride width.
     * @param[in]  upscaleX  Dilation height (vertical upscale).
     * @param[in]  upscaleY  Dilation width (horizontal upscale).
     *
     * @return     Convolution descriptor object.
     */
    static ConvolutionDescriptor create(
        int padH,
        int padW,
        int strideH,
        int strideW,
        int upscaleX = 1,
        int upscaleY = 1) {
        ConvolutionDescriptor object;
	checkStatus(
            cudnnSetConvolution2dDescriptor(
                object,
                padH,
                padW,
                strideH,
                strideW,
                upscaleX,
                upscaleY,
                CUDNN_CONVOLUTION,
		detail::dataType<T>::type));
        return object;
    }

    /**
     * @brief      Creates a new descriptor object for CuDNN cross-correlation.
     *
     * @param[in]  padH      Padding height.
     * @param[in]  padW      Padding width.
     * @param[in]  strideH   Stride height.
     * @param[in]  strideW   Stride withd.
     * @param[in]  upscaleX  Dilation height (vertical upscale).
     * @param[in]  upscaleY  Dilation width (horizontal upscale).
     *
     * @return     Cross-correlation object.
     */
     static ConvolutionDescriptor createCrossCorrelation(
        int padH,
        int padW,
        int strideH,
        int strideW,
        int upscaleX = 1,
        int upscaleY = 1) {
        ConvolutionDescriptor object;
        checkStatus(
            cudnnSetConvolution2dDescriptor(
                object,
                padH,
                padW,
                strideH,
                strideW,
                upscaleX,
                upscaleY,
                CUDNN_CROSS_CORRELATION,
		detail::dataType<T>::type));
        return object;
    }

//////////////////////////////////////////////////////////////////////////////
// CuDNN Dropout Descriptor
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN dropout descriptor class.
 */
class DropoutDescriptor :
    public detail::RAII<cudnnDropoutDescriptor_t,
                        cudnnCreateDropoutDescriptor,
                        cudnnDestroyDropoutDescriptor> {};
};

//////////////////////////////////////////////////////////////////////////////
// CuDNN Filter Descriptor
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN filter descriptor class.
 *
 * @tparam     T     Filter data type (float or double)
 */
template <typename T>
class FilterDescriptor :
    public detail::RAII<cudnnFilterDescriptor_t,
                        cudnnCreateFilterDescriptor,
                        cudnnDestroyFilterDescriptor> {
 protected:
    FilterDescriptor() {
        static_assert(  // make sure correct data type is provided
		      std::is_same<T, float>::value ||
		      std::is_same<T, double>::value,
		      "Invalid Cudnn::FilterDescriptor<T> data-type");
    }
};

template <typename T>
class Filter4dDescriptor : public FilterDescriptor<T> {
    /**
     * @brief      Constructs the filter descriptor object.
     *
     * @param[in]  format  Tensor format.
     * @param[in]  k       K-dimension.
     * @param[in]  c       C-dimension.
     * @param[in]  h       H-dimension.
     * @param[in]  w       W-dimension.
     */
    Filter4dDescriptor(TensorFormat format, int k, int c, int h, int w) :
        FilterDescriptor<T>() {
        auto type = detail::dataType<T>::type;
        checkStatus(
	    cudnnSetFilter4dDescriptor(*this, type, format, k, c, h, w));
    }

 public:
    /**
     * @brief      Creates a new filter descriptor object using NCHW tensor 
     *             format.
     *
     * @param[in]  k     K-dimension.
     * @param[in]  c     C-dimension.
     * @param[in]  h     H-dimension.
     * @param[in]  w     W-dimension.
     *
     * @return     NCHW filter descriptor object.
     */
    static Filter4dDescriptor createNCHW(int k, int c, int h, int w) {
        return Filter4dDescriptor(CUDNN_TENSOR_NCHW, k, c, h, w);
    }

    /**
     * @brief      Creates a new filter descriptor object using NHCW tensor
     *             format.
     *
     * @param[in]  k     K-dimension.
     * @param[in]  c     C-dimension.
     * @param[in]  h     H-dimension.
     * @param[in]  w     W-dimension.
     *
     * @return     NCHW filter descriptor object.
     */
    static Filter4dDescriptor createNHWC(int k, int c, int h, int w) {
        return Filter4dDescriptor(CUDNN_TENSOR_NHWC, k, c, h, w);
    }
};


//////////////////////////////////////////////////////////////////////////////
// CuDNN LRN Descriptor
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN LRN descriptor class.
 */
class LRNDescriptor :
    public detail::RAII<cudnnLRNDescriptor_t,
                        cudnnCreateLRNDescriptor,
                        cudnnDestroyLRNDescriptor> {};

//////////////////////////////////////////////////////////////////////////////
// CuDNN Operation Tensor Descriptor
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN operation tensor descriptor class.
 */
class OpTensorDescriptor :
    public detail::RAII<cudnnOpTensorDescriptor_t,
                        cudnnCreateOpTensorDescriptor,
                        cudnnDestroyOpTensorDescriptor> {};

//////////////////////////////////////////////////////////////////////////////
// CuDNN Pooling Descriptor
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN pooling descriptor class.
 */
class PoolingDescriptor :
    public detail::RAII<cudnnPoolingDescriptor_t,
                        cudnnCreatePoolingDescriptor,
                        cudnnDestroyPoolingDescriptor> {
    PoolingDescriptor() {}

 public:
    /**
     * @brief      Creates a new pooling descriptor object.
     *
     * @param[in]  winH     Window width.
     * @param[in]  winW     Window height.
     * @param[in]  padH     Padding width.
     * @param[in]  padW     Padding height.
     * @param[in]  hStride  Stride height.
     * @param[in]  wStride  Stride width.
     * @param[in]  mode     Pooling mode.
     *
     * @return     CuDNN pooling descriptor object.
     */
    static PoolingDescriptor create(int winH, int winW,
                                    int padH, int padW,
                                    int hStride, int wStride,
                                    cudnnPoolingMode_t mode) {
        PoolingDescriptor object;
        CuDNN::checkStatus(
            cudnnSetPooling2dDescriptor(
                object,
                mode,
                CUDNN_NOT_PROPAGATE_NAN,
                winH,
                winW,
                padH,
                padW,
                hStride,
                wStride));
        return object;
    }
};

//////////////////////////////////////////////////////////////////////////////
// CuDNN RNN Descriptor
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN RNN descriptor class.
 */
class RNNDescriptor :
    public detail::RAII<cudnnRNNDescriptor_t,
                        cudnnCreateRNNDescriptor,
                        cudnnDestroyRNNDescriptor> {};

//////////////////////////////////////////////////////////////////////////////
// CuDNN Spatial Transformer Descriptor
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN spatial transformer descriptor class.
 */
class SpatialTransformerDescriptor :
    public detail::RAII<cudnnSpatialTransformerDescriptor_t,
                        cudnnCreateSpatialTransformerDescriptor,
                        cudnnDestroySpatialTransformerDescriptor> {};

//////////////////////////////////////////////////////////////////////////////
// CuDNN Tensor Descriptor
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN tensor descriptor class.
 */
class TensorDescriptor :
    public detail::RAII<cudnnTensorDescriptor_t,
                        cudnnCreateTensorDescriptor,
                        cudnnDestroyTensorDescriptor> {};

//////////////////////////////////////////////////////////////////////////////
// CuDNN Convolution
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN convolution class.
 */
template <typename T>
class Convolution {

    Convolution() {
        static_assert(  // make sure correct data type is provided
		      std::is_same<T, float>::value ||
		      std::is_same<T, double>::value,
		      "Invalid CuDNN::Convolution<T> data-type");
    }

 public:
    /**
     * @brief      Creates a new workspace for forward convolution.
     * 
     * @param[in]  handle                CuDNN library handle.
     * @param[in]  inputDescriptor       Input tensor descriptor.
     * @param[in]  filterDescriptor      Filter descriptor.
     * @param[in]  convolutionDescritor  Convolution descriptor.
     * @param[in]  outputDescriptor      Output tensor descriptor.
     * @param[in]  algorithm             Forward convolution algorithm. 
     * 
     * @tparam     T                     Data type (float or double).
     * 
     * @return     Workspace buffer object.
     */
     detail::Buffer<T> static createForwardWorkspace(
        Handle handle,
        TensorDescriptor inputDescriptor,
        FilterDescriptor<T> filterDescriptor,
        ConvolutionDescriptor<T> convolutionDescritor,
        TensorDescriptor outputDescriptor,
        ConvolutionFwdAlgo algorithm) {
        size_t workspaceSize;
        CuDNN::checkStatus(
            cudnnGetConvolutionForwardWorkspaceSize(
                handle,
                inputDescriptor,
                filterDescriptor,
                convolutionDescritor,
                outputDescriptor,
                algorithm,
                &workspaceSize));
        return detail::Buffer<T>(workspaceSize);
    }

    /**
     * @brief      Creates a new workspace for backward data convolution.
     *
     * @param[in]  handle                CuDNN library handle.
     * @param[in]  diffInputDescriptor   Differential input tensor descriptor.
     * @param[in]  filterDescriptor      Filter descriptor.
     * @param[in]  convolutionDescritor  Convolution descriptor.
     * @param[in]  diffOutputDescriptor  Differential output tensor descriptor.
     * @param[in]  algorithm             Backward data convolution algorithm. 
     *
     * @tparam     T                     Data type (float or double).
     *
     * @return     Workspace buffer object.
     */
    detail::Buffer<T> createBackwardDataWorkspace(
        Handle handle,
        TensorDescriptor diffInputDescriptor,
        FilterDescriptor<T> filterDescriptor,
        ConvolutionDescriptor<T> convolutionDescritor,
        TensorDescriptor diffOutputDescriptor,
        ConvolutionBwdDataAlgo algorithm) {
        size_t workspaceSize;
        CuDNN::checkStatus(
            cudnnGetConvolutionBackwardDataWorkspaceSize(
                handle,
                filterDescriptor,
                diffOutputDescriptor,
                convolutionDescritor,
                diffInputDescriptor,
                algorithm,
                &workspaceSize));
        return detail::Buffer<T>(workspaceSize / sizeof(T));
    }

    /**
     * @brief      Creates a new workspace for backward filter convolution.
     *
     * @param[in]  handle                CuDNN library handle.
     * @param[in]  inputDescriptor       Input tensor descriptor.
     * @param[in]  filterDescriptor      Filter descriptor.
     * @param[in]  convolutionDescritor  Convolution descriptor.
     * @param[in]  diffOutputDescriptor  Differential output tensor descriptor.
     * @param[in]  algorithm             Backward filter convolution algorithm. 
     *
     * @tparam     T                     Data type (float or double).
     *
     * @return     Workspace buffer object.
     */
    detail::Buffer<T> createBackwardFilterWorkspace(
        Handle handle,
        TensorDescriptor inputDescriptor,
        FilterDescriptor<T> filterDescriptor,
        ConvolutionDescriptor<T> convolutionDescritor,
        TensorDescriptor diffOutputDescriptor,
        ConvolutionBwdFilterAlgo algorithm) {
        size_t workspaceSize;
        CuDNN::checkStatus(
            cudnnGetConvolutionBackwardFilterWorkspaceSize(
                handle,
                inputDescriptor,
                diffOutputDescriptor,
                convolutionDescritor,
                filterDescriptor,
                algorithm,
                &workspaceSize));
        return detail::Buffer<T>(workspaceSize / sizeof(T));
    }
};

//////////////////////////////////////////////////////////////////////////////
// CuDNN Filter
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN filter class.
 *
 * @tparam     T     The underlying data type (float or double)
 */
template <typename T>
class Filter {
    /**
     * Filter coefficients buffer.
     */
    detail::Buffer<T>               mBuffer;
    /**
     * Filter descriptor object.
     */
    FilterDescriptor<T>     mDescriptor;
    /**
     * @brief      Constructs the filter object.
     *
     * @param[in]  buffer      Filter coefficients buffer.
     * @param[in]  descriptor  Filter descriptor object.
     */
    Filter(const detail::Buffer<T>& buffer,
        const FilterDescriptor<T>& descriptor) :
        mBuffer(buffer),
        mDescriptor(descriptor) {}

 public:
    /**
     * @brief      Creates a new NCHW filter object.
     *
     * @param[in]  k     K-dimension.
     * @param[in]  c     C-dimension.
     * @param[in]  h     H-dimension.
     * @param[in]  w     W-dimension.
     *
     * @return     NCHW filter object.
     */
    static Filter createNCHW(int k, int c, int h, int w) {
        return Filter(detail::Buffer<T>(k * c * h * w),
            FilterDescriptor<T>::createNCHW(k, c, h, w));
    }
    /**
     * @brief      Creates a new NHWC filter object.
     *
     * @param[in]  k     K-dimension.
     * @param[in]  c     C-dimension.
     * @param[in]  h     H-dimension.
     * @param[in]  w     W-dimension.
     *
     * @return     NHWC filter object.
     */
    static Filter createNHWC(int k, int c, int h, int w) {
        return Filter(detail::Buffer<T>(k * c * h * w),
            FilterDescriptor<T>::createNHWC(k, c, h, w));
    }
    /**
     * @brief      Gets the filter descriptor.
     *
     * @return     Filter descritor object.
     */
    FilterDescriptor<T> getDescriptor() const {
        return mDescriptor;
    }
    /**
     * @brief      Defines conversion between Filter objects and 
     *             C-style pointers to the filter coefficients data. 
     */
    operator T*() const {
        return mBuffer;
    }
};

//////////////////////////////////////////////////////////////////////////////
// CuDNN Tensor
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief      CuDNN tensor class.
 *
 * @tparam     T     The underlying data type (float or double)
 */
template <typename T>
class Tensor {
    /**
     * Tensor data buffer.
     */
    detail::Buffer<T> mBuffer;
    /**
     * Tensor descriptor object.
     */
    CuDNN::TensorDescriptor mDescriptor;
    /**
     * Vecotr of tensor dimenstions.
     */
    std::vector<int> mDims;
    /**
     * @brief      Constructs the tensor object.
     *
     * @param[in]  n           N-dimension.
     * @param[in]  c           C-dimension.
     * @param[in]  h           H-dimension.
     * @param[in]  w           W-dimension.
     * @param[in]  descriptor  Tensor descriptor object.
     */
    Tensor(int n, int c, int h, int w, TensorDescriptor descriptor) :
        mBuffer(n * c * h * w),
        mDescriptor(descriptor),
        mDims({n, c, h, w}) {}
    /**
     * @brief      Constructs the tensor object.
     *
     * @param[in]  dims        Tensor dimensions
     * @param[in]  descriptor  Vector of tensor descriptor object.
     */
    Tensor(const std::vector<int> dims, TensorDescriptor descriptor) :
        mBuffer(dimsToSize(dims)),
        mDescriptor(descriptor),
        mDims(dims) {}

 private:
    /**
     * @brief      Helper function that calculates tensor size based on tensor
     *             dimensions.
     *
     * @param[in]  dims  Vector of tensor dimensions.
     *
     * @return     Size in number of elements (e.g.not number of bytes)
     */
    static size_t dimsToSize(const std::vector<int> dims) {
        return std::accumulate(dims.begin(), dims.end(), 1,
			       std::multiplies<int>());
    }
    /**
     * @brief      Factory function that creates a tensor descriptor object.
     *
     * @param[in]  descSetFunc   Pointer to cuDNN setter function.
     * @param[in]  args          Variadic list of arguments to pass into cuDNN
     *                           setter function. 
     *
     * @tparam     DecsSetTypes  Variadic list of cuDNN setter function argument 
     *                           types.
     * @tparam     ArgTypes      Variadic list of factory function argument 
     *                           types.
     *
     * @return     CuDNN tensor descriptor object.
     */
    template <typename ... DecsSetTypes, typename ... ArgTypes>
    static inline CuDNN::TensorDescriptor makeDescriptor(
        Status(*descSetFunc)(DecsSetTypes...), ArgTypes ... args) {
        CuDNN::TensorDescriptor descriptor;
        CuDNN::checkStatus(descSetFunc(descriptor, args ...));
        return descriptor;
    }

 public:
    /**
     * @brief      Defines conversion between Tensor objects and 
     *             C-style pointers to the underlying data. 
     */
    operator T*() const {
        return mBuffer;
    }
    /**
     * @brief      Gets the tensor descriptor.
     *
     * @return     CuDNN tensor descriptor object.
     */
    const TensorDescriptor& getDescriptor() const {
        return mDescriptor;
    }
    /**
     * @brief      Creates a new NCHW tensor object.
     *
     * @param[in]  n          N-dimension.
     * @param[in]  c          C-dimension.
     * @param[in]  h          H-dimension.
     * @param[in]  w          W-dimension.
     *
     * @return     CuDNN tensor object.
     */
    static Tensor<T> createNCHW(int n, int c, int h, int w) {
        return Tensor(n, c, h, w, makeDescriptor(
            &cudnnSetTensor4dDescriptor,
            CUDNN_TENSOR_NCHW,
            CuDNN::detail::dataType<T>::type,
            n, c, h, w));
    }
    /**
     * @brief      Creates a new NCHW tensor object.
     *
     * @param[in]  NCHW       Tuple of NCHW tensor dimensions.
     *
     * @return     CuDNN tensor object.
     */
    static Tensor<T> createNCHW(std::tuple<int, int, int, int> NCHW) {
        int n, c, h, w;
        std::tie(n, c, h, w) = NCHW;
        return Tensor(n, c, h, w, makeDescriptor(
            &cudnnSetTensor4dDescriptor,
            CUDNN_TENSOR_NCHW,
            CuDNN::detail::dataType<T>::type,
            n, c, h, w));
    }
    /**
     * @brief      Creates a new NHWC tensor object.
     *
     * @param[in]  n          N-dimension.
     * @param[in]  h          H-dimension.
     * @param[in]  w          W-dimension.
     * @param[in]  c          C-dimension.
     *
     * @return     CuDNN tensor object.
     */
    static Tensor<T> createNHWC(int n, int h, int w, int c) {
        return Tensor(n, h, w, c, makeDescriptor(
            &cudnnSetTensor4dDescriptor,
            CUDNN_TENSOR_NHWC,
            CuDNN::detail::dataType<T>::type,
            n, h, w, c));
    }
    /**
     * @brief      Creates a new NHWC tensor object.
     *
     * @param[in]  NHWC       Tuple of NHWC tensor dimensions.
     *
     * @return     CuDNN tensor object.
     */
    static Tensor<T> createNHWC(std::tuple<int, int, int, int> NHWC) {
        int n, h, w, c;
        std::tie(n, h, w, c) = NHWC;
        return Tensor(n, h, w, c, makeDescriptor(
            &cudnnSetTensor4dDescriptor,
            CUDNN_TENSOR_NHWC,
            CuDNN::detail::dataType<T>::type,
            n, h, w, c));
    }
    /**
     * @brief      Creates a new tensor object.
     *
     * @param[in]  dims       Vector of tensor dimensions.
     * @param[in]  strides    Vector of tensor strides.
     *
     * @return     CuDNN tensor object.
     */
    static Tensor<T> create(const std::vector<int>& dims,
                            const std::vector<int>& strides) {
        return Tensor(dims, makeDescriptor(
            &cudnnSetTensorNdDescriptor,
            CuDNN::detail::dataType<T>::type,
            dims.size(),
            dims.data(),
            strides.data()));
    }
};

}   // namespace CuDNN

#endif  // INCLUDE_CUDNN_HPP_
