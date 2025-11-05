#include <cuda_fp16.h>
#include <cuda_runtime.h>

// CUDA kernel for FP16 to FP32 conversion
__global__ void convertFP16toFP32Kernel(const __half* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}

// CUDA kernel for FP32 to FP16 conversion
__global__ void convertFP32toFP16Kernel(const float* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

// CUDA kernel to add regularization to diagonal
__global__ void addRegularizationKernel(float* matrices, int batchSize, int n, float eps) {
    int batch = blockIdx.x;
    int diag_idx = threadIdx.x;

    if (batch < batchSize && diag_idx < n) {
        int offset = batch * n * n + diag_idx * n + diag_idx;
        matrices[offset] += eps;
    }
}

// CUDA kernel to transpose from row-major to column-major for cuBLAS
__global__ void transposeRowMajorToColMajorKernel(const float* input, float* output, int batchSize, int rows, int cols) {
    int batch = blockIdx.x;
    int row = blockIdx.y;
    int col = threadIdx.x;

    if (batch < batchSize && row < rows && col < cols) {
        // Input is row-major: input[batch * rows * cols + row * cols + col]
        // Output should be column-major for cuBLAS: output[batch * rows * cols + col * rows + row]
        int input_idx = batch * rows * cols + row * cols + col;
        int output_idx = batch * rows * cols + col * rows + row;

        output[output_idx] = input[input_idx];
    }
}

// Host wrappers
extern "C" {

void launchConvertFP16toFP32(const __half* input, float* output, int size, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    convertFP16toFP32Kernel<<<gridSize, blockSize, 0, stream>>>(input, output, size);
}

void launchConvertFP32toFP16(const float* input, __half* output, int size, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    convertFP32toFP16Kernel<<<gridSize, blockSize, 0, stream>>>(input, output, size);
}

void launchAddRegularization(float* matrices, int batchSize, int n, float eps, cudaStream_t stream) {
    // Each block handles one batch, threads handle diagonal elements
    int blockSize = (n + 31) & ~31;  // Round up to multiple of 32 (warp size)
    if (blockSize > 1024) blockSize = 1024;  // Max threads per block
    addRegularizationKernel<<<batchSize, blockSize, 0, stream>>>(matrices, batchSize, n, eps);
}

void launchTransposeRowMajorToColMajor(const float* input, float* output, int batchSize, int rows, int cols, cudaStream_t stream) {
    // Use 2D grid: blockIdx.x = batch, blockIdx.y = row, threadIdx.x = col
    dim3 dimGrid(batchSize, rows);
    int blockSize = (cols + 31) & ~31;  // Round up to warp size
    if (blockSize > 1024) blockSize = 1024;

    transposeRowMajorToColMajorKernel<<<dimGrid, blockSize, 0, stream>>>(input, output, batchSize, rows, cols);
}

} // extern "C"