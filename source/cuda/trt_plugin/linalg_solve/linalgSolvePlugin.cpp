#include "linalgSolvePlugin.h"
#include "linalgSolveKernels.h"
#include "common.h"
#include "logger.h"
#include <cuda.h>
#include <cuda_fp16.h>

using namespace nvinfer1;

#define CUDRIVER_CALL(call)                                                                                            \
    {                                                                                                                  \
        cudaError_enum s_ = call;                                                                                      \
        if (s_ != CUDA_SUCCESS)                                                                                        \
        {                                                                                                              \
            char const *errName_, *errDesc_;                                                                           \
            cuGetErrorName(s_, &errName_);                                                                             \
            cuGetErrorString(s_, &errDesc_);                                                                           \
            gLogError << "CUDA Error: " << errName_ << " " << errDesc_ << std::endl;                           \
            return s_;                                                                                                 \
        }                                                                                                              \
    }

#define CUDA_CALL(call)                                                                                                \
    {                                                                                                                  \
        cudaError_t s_ = call;                                                                                         \
        if (s_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            gLogError << "CUDA Error: " << cudaGetErrorName(s_) << " " << cudaGetErrorString(s_) << std::endl; \
            return s_;                                                                                                 \
        }                                                                                                              \
    }

#define CUBLAS_CALL(call)                                                                                              \
    {                                                                                                                  \
        cublasStatus_t s_ = call;                                                                                      \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                                               \
        {                                                                                                              \
            gLogError << "cuBLAS Error: " << s_ << std::endl;                                                        \
            return s_;                                                                                                 \
        }                                                                                                              \
    }

    // Helper function for serializing plugin
template <typename T>
void writeToBuffer(uint8_t*& buffer, T const& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(uint8_t const*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

REGISTER_TENSORRT_PLUGIN(LinalgSolvePluginCreator);

namespace
{
static const char* LINALG_SOLVE_PLUGIN_VERSION{"1"};
static const char* LINALG_SOLVE_PLUGIN_NAME{"LinalgSolvePlugin"};
}


// ============================================================================
// LinalgSolvePlugin Implementation
// ============================================================================

LinalgSolvePlugin::LinalgSolvePlugin(int32_t qPlusOne, float epsilon, float eta)
    : mQp1(qPlusOne), mEps(epsilon), mEta(eta)
{
    mCublas = nullptr;
    d_A_ptrs = nullptr;
    d_B_ptrs = nullptr;
    d_info = nullptr;
    d_pivots = nullptr;
}
    
LinalgSolvePlugin::LinalgSolvePlugin(void const* data, size_t length)
{
    // Deserialize mQp1 and mEps
    const char* d = reinterpret_cast<const char*>(data);
    std::memcpy(&mQp1, d, sizeof(mQp1));
    d += sizeof(mQp1);
    std::memcpy(&mEps, d, sizeof(mEps));
    d += sizeof(mEps);
    std::memcpy(&mEta, d, sizeof(mEta));
    d += sizeof(mEta);

    mCublas = nullptr;
    d_A_ptrs = nullptr;
    d_B_ptrs = nullptr;
    d_info = nullptr;
    d_pivots = nullptr;
}

LinalgSolvePlugin::~LinalgSolvePlugin()
{
    if (d_A_ptrs) cudaFree(d_A_ptrs);
    if (d_B_ptrs) cudaFree(d_B_ptrs);
    if (d_info) cudaFree(d_info);
    if (d_pivots) cudaFree(d_pivots);
    if (mCublas) cublasDestroy(mCublas);
}

int32_t LinalgSolvePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t LinalgSolvePlugin::initialize() noexcept
{
    return 0;
}

const char* LinalgSolvePlugin::getPluginType() const noexcept {
    return LINALG_SOLVE_PLUGIN_NAME;
}

char const* LinalgSolvePlugin::getPluginVersion() const noexcept
{
    return LINALG_SOLVE_PLUGIN_VERSION;
}

DimsExprs LinalgSolvePlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs,
                                                  int nbInputs, IExprBuilder& exprBuilder) noexcept {
    // Input: XtX [B, H, W, (Q+1)*(Q+1)], XtY [B, H, W, (Q+1)*C]
    // Output: coeffs [B, H, W, (Q+1)*C] (same shape as XtY)
    assert(nbInputs == 2);
    assert(outputIndex == 0);
    return inputs[1]; // Same shape as XtY
}

void LinalgSolvePlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
    cublasStatus_t ret = cublasCreate(&mCublas);
    ASSERT(ret == CUBLAS_STATUS_SUCCESS && mCublas != nullptr && "Failed to create cublasHandle_t.");
}

// Detach the plugin object from its execution context.
void LinalgSolvePlugin::detachFromContext() noexcept {}

int LinalgSolvePlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                               const void* const* inputs, void* const* outputs,
                               void* workspace, cudaStream_t stream) noexcept {
    ASSERT(inputDesc != nullptr
        && outputDesc != nullptr
        && inputs != nullptr
        && outputs != nullptr);

    // Create cuBLAS stream
    CUBLAS_CALL(cublasSetStream(mCublas, stream));

    // Input 0: XTX [B, H, W, (Q+1)^2]
    // Input 1: XTY [B, H, W, (Q+1)*C]
    int B = inputDesc[0].dims.d[0];
    int H = inputDesc[0].dims.d[1];
    int W = inputDesc[0].dims.d[2];
    int Q_plus_1_sq = inputDesc[0].dims.d[3];
    int Q_plus_1 = static_cast<int>(std::sqrt(static_cast<float>(Q_plus_1_sq)));
    int C = inputDesc[1].dims.d[3] / Q_plus_1;

    // Cublas Batch
    int batchSize = B * H * W;
    bool isFP16 = (inputDesc[0].type == DataType::kHALF);

    // Get input/output pointers (could be FP16 or FP32)
    const void* d_XTX_input = inputs[0];
    const void* d_XTY_input = inputs[1];
    void* d_output_final = outputs[0];

    // Allocate temporary buffers if needed or size changed
    if (mBatch != B || mH != H || mW != W || mC != C) {
        // Free old buffers
        if (d_A_ptrs) { cudaFree(d_A_ptrs); d_A_ptrs = nullptr; }
        if (d_B_ptrs) { cudaFree(d_B_ptrs); d_B_ptrs = nullptr; }
        if (d_info) { cudaFree(d_info); d_info = nullptr; }
        if (d_pivots) { cudaFree(d_pivots); d_pivots = nullptr; }

        // Allocate new buffers
        CUDA_CALL(cudaMalloc(&d_A_ptrs, batchSize * sizeof(float*)));
        CUDA_CALL(cudaMalloc(&d_B_ptrs, batchSize * sizeof(float*)));
        CUDA_CALL(cudaMalloc(&d_info, batchSize * sizeof(int)));
        CUDA_CALL(cudaMalloc(&d_pivots, batchSize * Q_plus_1 * sizeof(int)));

        // Cache dimensions
        mBatch = B;
        mH = H;
        mW = W;
        mC = C;
    }

    // Reshape XTX from [B, H, W, (Q+1)^2] to [B*H*W, Q+1, Q+1]

    // Reshape XTY from [B, H, W, (Q+1)*C] to [B*H*W, Q+1, C]


    // For cuBLAS batched solver, we need to:
    // 1. Add regularization: XTX_reg = XTX + eps * I
    // 2. Solve: A = XTX_reg^(-1) * XTY using cublasSgetrfBatched + cublasSgetrsBatched

    // Since TensorRT input layout is [B, H, W, C], we need to be careful with memory layout
    // The last dimension is contiguous, so XTX at position [b,h,w,:] is contiguous

    // Use workspace for temporary buffers (avoid cudaMalloc/cudaFree in hot path)
    char* workspace_ptr = static_cast<char*>(workspace);
    size_t offset = 0;

    // Allocate from workspace
    float* d_XTX_reg = reinterpret_cast<float*>(workspace_ptr + offset);
    offset += batchSize * Q_plus_1 * Q_plus_1 * sizeof(float);

    float* d_XTY_copy = reinterpret_cast<float*>(workspace_ptr + offset);
    offset += batchSize * Q_plus_1 * C * sizeof(float);

    float* d_XTY_transposed = reinterpret_cast<float*>(workspace_ptr + offset);
    offset += batchSize * Q_plus_1 * C * sizeof(float);

    float* d_XTX_fp32 = nullptr;
    float* d_XTY_fp32 = nullptr;

    // Convert inputs to FP32 if they are FP16
    if (isFP16) {
        int totalXTXSize = batchSize * Q_plus_1 * Q_plus_1;
        int totalXTYSize = batchSize * Q_plus_1 * C;

        // Allocate FP16->FP32 conversion buffers from workspace
        d_XTX_fp32 = reinterpret_cast<float*>(workspace_ptr + offset);
        offset += totalXTXSize * sizeof(float);

        d_XTY_fp32 = reinterpret_cast<float*>(workspace_ptr + offset);
        offset += totalXTYSize * sizeof(float);

        // Convert XTX from FP16 to FP32
        launchConvertFP16toFP32(static_cast<const __half*>(d_XTX_input), d_XTX_fp32, totalXTXSize, stream);

        // Convert XTY from FP16 to FP32
        launchConvertFP16toFP32(static_cast<const __half*>(d_XTY_input), d_XTY_fp32, totalXTYSize, stream);

        // Copy converted data to workspace
        CUDA_CALL(cudaMemcpyAsync(d_XTX_reg, d_XTX_fp32, totalXTXSize * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));
        CUDA_CALL(cudaMemcpyAsync(d_XTY_copy, d_XTY_fp32, totalXTYSize * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));
    } else {
        // Direct copy for FP32 inputs
        CUDA_CALL(cudaMemcpyAsync(d_XTX_reg, d_XTX_input,
                                  batchSize * Q_plus_1 * Q_plus_1 * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));
        CUDA_CALL(cudaMemcpyAsync(d_XTY_copy, d_XTY_input,
                                  batchSize * Q_plus_1 * C * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));
    }

    // Add regularization (eps * I) to diagonal of each XTX matrix
    launchAddRegularization(d_XTX_reg, batchSize, Q_plus_1, mEps, stream);

    // Transpose XTY from row-major to column-major for cuBLAS
    // Input: [batch, Q+1, C] row-major
    // Output: [batch, Q+1, C] column-major (each matrix is transposed within batch)
    launchTransposeRowMajorToColMajor(d_XTY_copy, d_XTY_transposed, batchSize, Q_plus_1, C, stream);

    // Synchronize to ensure regularization and transpose are complete before cuBLAS operations
    CUDA_CALL(cudaStreamSynchronize(stream));

    // Use cuBLAS batched solver (optimized version)
    // Create array of pointers for batched operations
    std::vector<float*> h_A_ptrs(batchSize);
    std::vector<float*> h_B_ptrs(batchSize);
    for (int i = 0; i < batchSize; ++i) {
        h_A_ptrs[i] = d_XTX_reg + i * Q_plus_1 * Q_plus_1;
        h_B_ptrs[i] = d_XTY_transposed + i * Q_plus_1 * C;  // Use transposed data
    }

    // Copy pointer arrays to device asynchronously
    CUDA_CALL(cudaMemcpyAsync(d_A_ptrs, h_A_ptrs.data(), batchSize * sizeof(float*),
                              cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMemcpyAsync(d_B_ptrs, h_B_ptrs.data(), batchSize * sizeof(float*),
                              cudaMemcpyHostToDevice, stream));

    // Perform LU factorization
    CUBLAS_CALL(cublasSgetrfBatched(mCublas, Q_plus_1,
                                    reinterpret_cast<float**>(d_A_ptrs),
                                    Q_plus_1,
                                    static_cast<int*>(d_pivots),
                                    static_cast<int*>(d_info),
                                    batchSize));

    // IMPORTANT: Solve ALL RHS columns at once, not in a loop!
    int h_info_getrs = 0;  // Use host info, not device info
    CUBLAS_CALL(cublasSgetrsBatched(mCublas, CUBLAS_OP_N, Q_plus_1, C,
                                    reinterpret_cast<const float**>(d_A_ptrs),
                                    Q_plus_1,
                                    static_cast<const int*>(d_pivots),
                                    reinterpret_cast<float**>(d_B_ptrs),
                                    Q_plus_1,
                                    &h_info_getrs,  // Host pointer for info
                                    batchSize));

    // Transpose result back from column-major to row-major
    // d_XTY_transposed contains column-major result, transpose to d_XTY_copy
    launchTransposeRowMajorToColMajor(d_XTY_transposed, d_XTY_copy, batchSize, C, Q_plus_1, stream);

    // Convert result back to FP16 if needed and copy to output
    if (isFP16) {
        int totalOutputSize = batchSize * Q_plus_1 * C;
        launchConvertFP32toFP16(d_XTY_copy, static_cast<__half*>(d_output_final), totalOutputSize, stream);
    } else {
        // Direct copy for FP32 output
        CUDA_CALL(cudaMemcpyAsync(d_output_final, d_XTY_copy,
                                  batchSize * Q_plus_1 * C * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));
    }

    // No need to free workspace buffers - they are managed by TensorRT

    return 0;
}

IPluginV2DynamicExt* LinalgSolvePlugin::clone() const noexcept {
    auto* plugin = new LinalgSolvePlugin(mQp1, mEps, mEta);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}


bool LinalgSolvePlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut,
                                                   int nbInputs, int nbOutputs) noexcept {
    assert(nbInputs == 2 && nbOutputs == 1);
    assert(pos < 3);

    // Support FP32 (FP16 will be converted to FP32 internally for computation)
    bool isValidFormat = (inOut[pos].format == TensorFormat::kLINEAR ||
                          inOut[pos].format == TensorFormat::kHWC8) &&
                         (inOut[pos].type == DataType::kFLOAT ||
                          inOut[pos].type == DataType::kHALF);

    // All tensors must have the same data type
    if (pos > 0) {
        isValidFormat = isValidFormat && (inOut[pos].type == inOut[0].type);
    }

    return isValidFormat;
}

void LinalgSolvePlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
                                        const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
    assert(nbInputs == 2 && nbOutputs == 1);
}

size_t LinalgSolvePlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
                                           const PluginTensorDesc* outputs, int nbOutputs) const noexcept {
    // Calculate workspace needed for FP16->FP32 conversion and intermediate buffers
    int B = inputs[0].dims.d[0];
    int H = inputs[0].dims.d[1];
    int W = inputs[0].dims.d[2];
    int Q_plus_1_sq = inputs[0].dims.d[3];
    int Q_plus_1 = static_cast<int>(std::sqrt(static_cast<float>(Q_plus_1_sq)));
    int C = inputs[1].dims.d[3] / Q_plus_1;
    int batchSize = B * H * W;

    size_t workspace = 0;

    // Space for XTX_reg, XTY_copy, and XTY_transposed (always FP32)
    workspace += batchSize * Q_plus_1 * Q_plus_1 * sizeof(float); // XTX_reg
    workspace += batchSize * Q_plus_1 * C * sizeof(float);          // XTY_copy
    workspace += batchSize * Q_plus_1 * C * sizeof(float);          // XTY_transposed

    // If FP16 input, need space for FP32 conversion
    if (inputs[0].type == DataType::kHALF) {
        workspace += batchSize * Q_plus_1 * Q_plus_1 * sizeof(float); // XTX_fp32
        workspace += batchSize * Q_plus_1 * C * sizeof(float);          // XTY_fp32
    }

    return workspace;
}



DataType LinalgSolvePlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept {
    return inputTypes[0];
}

void LinalgSolvePlugin::terminate() noexcept {
}

size_t LinalgSolvePlugin::getSerializationSize() const noexcept {
    return sizeof(mQp1) + sizeof(mEps) + sizeof(mEta);
}

void LinalgSolvePlugin::serialize(void* buffer) const noexcept {
    uint8_t* d = static_cast<uint8_t*>(buffer);
    uint8_t* const a = d;
    writeToBuffer(d, mQp1);
    writeToBuffer(d, mEps);
    writeToBuffer(d, mEta);

    ASSERT(d == a + getSerializationSize());
}

void LinalgSolvePlugin::destroy() noexcept {
    delete this;
}

void LinalgSolvePlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* LinalgSolvePlugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

// ============================================================================
// LinalgSolvePluginCreator Implementation
// ============================================================================

LinalgSolvePluginCreator::LinalgSolvePluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("q_plus_one", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("eta", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LinalgSolvePluginCreator::getPluginName() const noexcept {
    return LINALG_SOLVE_PLUGIN_NAME;
}

const char* LinalgSolvePluginCreator::getPluginVersion() const noexcept {
    return LINALG_SOLVE_PLUGIN_VERSION;
}

const PluginFieldCollection* LinalgSolvePluginCreator::getFieldNames() noexcept {
    return &mFC;
}

IPluginV2DynamicExt* LinalgSolvePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    int32_t qPlusOne = 3;    // Default value
    float epsilon = 0.01f;   // Default value
    float eta = 0.001f;      // Default value

    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attrName = fc->fields[i].name;
        if (!strcmp(attrName, "q_plus_one")) {
            qPlusOne = *(static_cast<const int32_t*>(fc->fields[i].data));
        } else if (!strcmp(attrName, "epsilon")) {
            epsilon = *(static_cast<const float*>(fc->fields[i].data));
        } else if (!strcmp(attrName, "eta")) {
            eta = *(static_cast<const float*>(fc->fields[i].data));
        }
    }

    auto* plugin = new LinalgSolvePlugin(qPlusOne, epsilon, eta);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* LinalgSolvePluginCreator::deserializePlugin(const char* name, const void* serialData,
                                                       size_t serialLength) noexcept {
    auto* plugin = new LinalgSolvePlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void LinalgSolvePluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* LinalgSolvePluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

