#ifndef GAUSSIAN_BLUR_KERNELS_H
#define GAUSSIAN_BLUR_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// Launch FP32 Gaussian blur kernel for TensorRT plugin
// Input/Output layout: NCHW (batch_size, channels, height, width)
void launchGaussianBlurFP32(
    const float* input,     // Input tensor [N, C, H, W]
    float* output,          // Output tensor [N, C, H, W]
    float* workspace,       // Workspace for intermediate result [N, C, H, W]
    int batch_size,         // Batch size (N)
    int channels,           // Number of channels (C)
    int height,             // Image height (H)
    int width,              // Image width (W)
    float sigma,            // Gaussian sigma parameter
    cudaStream_t stream     // CUDA stream
);

// Launch FP16 Gaussian blur kernel for TensorRT plugin
// Input/Output layout: NCHW (batch_size, channels, height, width)
void launchGaussianBlurFP16(
    const __half* input,    // Input tensor [N, C, H, W]
    __half* output,         // Output tensor [N, C, H, W]
    __half* workspace,      // Workspace for intermediate result [N, C, H, W]
    int batch_size,         // Batch size (N)
    int channels,           // Number of channels (C)
    int height,             // Image height (H)
    int width,              // Image width (W)
    float sigma,            // Gaussian sigma parameter
    cudaStream_t stream     // CUDA stream
);

#ifdef __cplusplus
}
#endif

#endif // GAUSSIAN_BLUR_KERNELS_H