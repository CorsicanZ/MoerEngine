#ifndef LINALG_SOLVE_KERNELS_H
#define LINALG_SOLVE_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

// Launch FP16 to FP32 conversion kernel
void launchConvertFP16toFP32(const __half* input, float* output, int size, cudaStream_t stream);

// Launch FP32 to FP16 conversion kernel
void launchConvertFP32toFP16(const float* input, __half* output, int size, cudaStream_t stream);

// Launch regularization kernel
void launchAddRegularization(float* matrices, int batchSize, int n, float eps, cudaStream_t stream);

// Launch transpose kernel to convert row-major to column-major for cuBLAS
void launchTransposeRowMajorToColMajor(const float* input, float* output, int batchSize, int rows, int cols, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // LINALG_SOLVE_KERNELS_H