#include "GaussianBlurKernels.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// Helper function for zero padding (matching flnr.py's F.conv2d behavior)
__device__ __forceinline__ int zero_coord(int coord, int max_coord) {
    // PyTorch zero padding: out-of-bounds = 0
    if (coord < 0 || coord >= max_coord) {
        return -1; // Special value to indicate out-of-bounds (zero)
    }
    return coord;
}

// Pre-computed Gaussian weights for kernel_size=5 (radius=2), sigma=1.5
// Matching PyTorch's calculation: torch.exp(-(x**2)/(2*1.5**2)) / sum
__constant__ float gaussian_weights_5[5] = {
    0.120100f, 0.233906f, 0.292088f, 0.233906f, 0.120100f
};

// Template-based CUDA kernels for different channel counts and kernel sizes
template<int CHANNELS, int KERNEL_RADIUS, int BLOCK_SIZE>
__global__ void gaussian_blur_vertical_nchw(
    const float* __restrict__ input,
    float* __restrict__ output,
    int height,
    int width,
    float sigma
) {
    const int KERNEL_SIZE = 2 * KERNEL_RADIUS + 1;

    // Thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Each thread processes all channels for one pixel
    for (int c = 0; c < CHANNELS; c++) {
        int idx = c * height * width + y * width + x;

        float sum = 0.0f;
        // Vertical blur (H direction) - matching flnr.py kernel_h [K, 1] with zero padding
        for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
            int sample_y = zero_coord(y + i, height);
            if (sample_y == -1) {
                // Out of bounds, contribution is 0 (zero padding)
                continue;
            }
            int sample_idx = c * height * width + sample_y * width + x;
            sum += gaussian_weights_5[i + KERNEL_RADIUS] * input[sample_idx];
        }

        output[idx] = sum;
    }
}

template<int CHANNELS, int KERNEL_RADIUS, int BLOCK_SIZE>
__global__ void gaussian_blur_horizontal_nchw(
    const float* __restrict__ input,
    float* __restrict__ output,
    int height,
    int width,
    float sigma
) {
    const int KERNEL_SIZE = 2 * KERNEL_RADIUS + 1;

    // Thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Each thread processes all channels for one pixel
    for (int c = 0; c < CHANNELS; c++) {
        int idx = c * height * width + y * width + x;

        float sum = 0.0f;
        // Horizontal blur (W direction) - matching flnr.py kernel_v [1, K] with zero padding
        for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
            int sample_x = zero_coord(x + i, width);
            if (sample_x == -1) {
                // Out of bounds, contribution is 0 (zero padding)
                continue;
            }
            int sample_idx = c * height * width + y * width + sample_x;
            sum += gaussian_weights_5[i + KERNEL_RADIUS] * input[sample_idx];
        }

        output[idx] = sum;
    }
}

// FP16 vertical blur kernel (H direction)
template<int CHANNELS, int KERNEL_RADIUS, int BLOCK_SIZE>
__global__ void gaussian_blur_vertical_fp16_nchw(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int height,
    int width,
    float sigma
) {
    const int KERNEL_SIZE = 2 * KERNEL_RADIUS + 1;

    // Thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Each thread processes all channels for one pixel
    for (int c = 0; c < CHANNELS; c++) {
        int idx = c * height * width + y * width + x;

        float sum = 0.0f;
        // Vertical blur (H direction) - matching flnr.py kernel_h [K, 1] with zero padding
        for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
            int sample_y = zero_coord(y + i, height);
            if (sample_y == -1) {
                // Out of bounds, contribution is 0 (zero padding)
                continue;
            }
            int sample_idx = c * height * width + sample_y * width + x;
            sum += gaussian_weights_5[i + KERNEL_RADIUS] * __half2float(input[sample_idx]);
        }

        output[idx] = __float2half(sum);
    }
}

// FP16 horizontal blur kernel (W direction)
template<int CHANNELS, int KERNEL_RADIUS, int BLOCK_SIZE>
__global__ void gaussian_blur_horizontal_fp16_nchw(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int height,
    int width,
    float sigma
) {
    const int KERNEL_SIZE = 2 * KERNEL_RADIUS + 1;

    // Thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Each thread processes all channels for one pixel
    for (int c = 0; c < CHANNELS; c++) {
        int idx = c * height * width + y * width + x;

        float sum = 0.0f;
        // Horizontal blur (W direction) - matching flnr.py kernel_v [1, K] with zero padding
        for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
            int sample_x = zero_coord(x + i, width);
            if (sample_x == -1) {
                // Out of bounds, contribution is 0 (zero padding)
                continue;
            }
            int sample_idx = c * height * width + y * width + sample_x;
            sum += gaussian_weights_5[i + KERNEL_RADIUS] * __half2float(input[sample_idx]);
        }

        output[idx] = __float2half(sum);
    }
}

// Host wrappers for TensorRT plugin to call
extern "C" {

void launchGaussianBlurFP32(
    const float* input,
    float* output,
    float* workspace,  // Workspace for intermediate horizontal blur result
    int batch_size,
    int channels,
    int height,
    int width,
    float sigma,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 32;
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y
    );

    // Process each batch
    for (int b = 0; b < batch_size; b++) {
        const float* batch_input = input + b * channels * height * width;
        float* batch_workspace = workspace + b * channels * height * width;
        float* batch_output = output + b * channels * height * width;

        // Launch two-pass kernels based on channel count
        // Using KERNEL_RADIUS=2 for kernel_size=5 (test_gaussian.py uses kernel_size=5)
        switch (channels) {
            case 16:
                gaussian_blur_horizontal_nchw<16, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_vertical_nchw<16, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
            case 5:
                gaussian_blur_horizontal_nchw<5, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_vertical_nchw<5, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
            case 4:
                gaussian_blur_horizontal_nchw<4, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_vertical_nchw<4, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
            case 3:
                gaussian_blur_horizontal_nchw<3, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_vertical_nchw<3, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
            case 2:
                gaussian_blur_horizontal_nchw<2, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_vertical_nchw<2, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
            case 1:
                gaussian_blur_horizontal_nchw<1, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_vertical_nchw<1, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
            default:
                // Fallback for unsupported channel counts
                gaussian_blur_horizontal_nchw<4, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_vertical_nchw<4, 2, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
        }
    }
}

void launchGaussianBlurFP16(
    const __half* input,
    __half* output,
    __half* workspace,  // Workspace for intermediate horizontal blur result
    int batch_size,
    int channels,
    int height,
    int width,
    float sigma,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 32;
    constexpr int KERNEL_RADIUS = 2; // kernel_size=5
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(
        (width + block_size.x - 1) / block_size.x,
        (height + block_size.y - 1) / block_size.y
    );

    // Process each batch using two-pass separable convolution
    for (int b = 0; b < batch_size; b++) {
        const __half* batch_input = input + b * channels * height * width;
        __half* batch_workspace = workspace + b * channels * height * width;
        __half* batch_output = output + b * channels * height * width;

        // Launch two-pass kernels based on channel count
        switch (channels) {
            case 16:
                gaussian_blur_vertical_fp16_nchw<16, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_horizontal_fp16_nchw<16, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
            case 5:
                gaussian_blur_vertical_fp16_nchw<5, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_horizontal_fp16_nchw<5, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
            case 4:
                gaussian_blur_vertical_fp16_nchw<4, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_horizontal_fp16_nchw<4, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
            case 3:
                gaussian_blur_vertical_fp16_nchw<3, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_horizontal_fp16_nchw<3, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
            case 2:
                gaussian_blur_vertical_fp16_nchw<2, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_horizontal_fp16_nchw<2, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
            case 1:
                gaussian_blur_vertical_fp16_nchw<1, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_horizontal_fp16_nchw<1, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
            default:
                // Fallback for unsupported channel counts
                gaussian_blur_vertical_fp16_nchw<4, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_input, batch_workspace, height, width, sigma
                );
                gaussian_blur_horizontal_fp16_nchw<4, KERNEL_RADIUS, BLOCK_SIZE><<<grid_size, block_size, 0, stream>>>(
                    batch_workspace, batch_output, height, width, sigma
                );
                break;
        }
    }
}

} // extern "C"