# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(FLNRPlugins LANGUAGES CXX CUDA)

if(NOT MSVC)
    # Enable all compile warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
endif()

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if(NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
if(MSVC)
    # Windows paths
    set_ifndef(TRT_LIB "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/lib")
    set_ifndef(TRT_INCLUDE "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/include")
    set_ifndef(CUDA_INC_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include")
    set_ifndef(CUDA_LIB_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
    set_ifndef(CUBLAS_LIB_SUFFIXES "lib;lib64")
else()
    # Linux paths
    set_ifndef(TRT_LIB /usr/lib/x86_64-linux-gnu)
    set_ifndef(TRT_INCLUDE /usr/include/x86_64-linux-gnu)
    set_ifndef(CUDA_INC_DIR /usr/local/cuda/include)
    set_ifndef(CUDA_LIB_DIR /usr/local/cuda)
    set_ifndef(CUBLAS_LIB_SUFFIXES "lib;lib64")
endif()

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
if(MSVC)
    set(NVINFER_LIB "${TRT_LIB}/nvinfer_10.lib")
    set(CUDART_LIB "${CUDA_LIB_DIR}/lib/x64/cudart.lib")
    set(CUBLAS_LIB "${CUDA_LIB_DIR}/lib/x64/cublas.lib")
    set(CUDA_LIB "${CUDA_LIB_DIR}/lib/x64/cuda.lib")
else()
    find_library(
        _NVINFER_LIB nvinfer
        HINTS ${TRT_LIB}
        PATH_SUFFIXES lib lib64)
    set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

    find_library(
        _CUDART_LIB cudart
        HINTS ${CUDA_LIB_DIR}
        PATH_SUFFIXES lib lib64)
    set_ifndef(CUDART_LIB ${_CUDART_LIB})

    find_library(
        _CUBLAS_LIB cublas
        HINTS ${CUDA_LIB_DIR}
        PATH_SUFFIXES ${CUBLAS_LIB_SUFFIXES})
    set_ifndef(CUBLAS_LIB ${_CUBLAS_LIB})

    find_library(
        _CUDA_LIB cuda
        HINTS ${CUDA_LIB_DIR}
        PATH_SUFFIXES lib/stubs lib64/stubs)
    set_ifndef(CUDA_LIB ${_CUDA_LIB})
endif()

# -------- BUILDING --------

add_definitions(-DTENSORRT_BUILD_LIB)

# Add include directories (shared include for common headers)
include_directories(${CUDA_INC_DIR} ${TRT_INCLUDE} ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/linalg_solve ${CMAKE_SOURCE_DIR}/gaussian_blur)

# -------- LinalgSolve Plugin --------
add_library(
    linalgSolvePlugin MODULE
    ${CMAKE_SOURCE_DIR}/linalg_solve/linalgSolvePlugin.cpp
    ${CMAKE_SOURCE_DIR}/linalg_solve/linalgSolvePlugin.h
    ${CMAKE_SOURCE_DIR}/linalg_solve/linalgSolveKernels.cu
    ${CMAKE_SOURCE_DIR}/linalg_solve/linalgSolveKernels.h
    ${CMAKE_SOURCE_DIR}/include/common.h
    ${CMAKE_SOURCE_DIR}/include/logger.h)

# Use C++17
target_compile_features(linalgSolvePlugin PUBLIC cxx_std_17)

# Link TensorRT's nvinfer lib
target_link_libraries(linalgSolvePlugin PRIVATE ${NVINFER_LIB})
target_link_libraries(linalgSolvePlugin PRIVATE ${CUDART_LIB})
target_link_libraries(linalgSolvePlugin PRIVATE ${CUBLAS_LIB})
target_link_libraries(linalgSolvePlugin PRIVATE ${CUDA_LIB})

# -------- GaussianBlur Plugin --------
add_library(
    gaussianBlurPlugin MODULE
    ${CMAKE_SOURCE_DIR}/gaussian_blur/GaussianBlurPlugin.cpp
    ${CMAKE_SOURCE_DIR}/gaussian_blur/GaussianBlurPlugin.h
    ${CMAKE_SOURCE_DIR}/gaussian_blur/GaussianBlurKernels.cu
    ${CMAKE_SOURCE_DIR}/gaussian_blur/GaussianBlurKernels.h)

# Use C++17
target_compile_features(gaussianBlurPlugin PUBLIC cxx_std_17)

# Link TensorRT's nvinfer lib
target_link_libraries(gaussianBlurPlugin PRIVATE ${NVINFER_LIB})
target_link_libraries(gaussianBlurPlugin PRIVATE ${CUDART_LIB})