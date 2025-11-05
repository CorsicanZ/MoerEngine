#ifndef LINALG_SOLVE_PLUGIN_H
#define LINALG_SOLVE_PLUGIN_H

#include "NvInferPlugin.h"
#include <cublas_v2.h>
#include <string>
#include <vector>

#include "MoerCudaAPI.h"

// LinalgSolve Plugin for TensorRT
// Solves regularized linear system for regression coefficients
class MOER_CUDA_API LinalgSolvePlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    LinalgSolvePlugin() = delete;
    LinalgSolvePlugin(int32_t qPlusOne, float epsilon, float eta);
    LinalgSolvePlugin(const void* data, size_t length);
    ~LinalgSolvePlugin() override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t                    index,
        nvinfer1::DimsExprs const* inputs,
        int32_t                    nbInputDims,
        nvinfer1::IExprBuilder&    exprBuilder
    ) noexcept override; // determine output dims based on input info

    template<typename TDataType>
    TDataType const* pointer_const_cast(void const* const p);

    template<typename TDataType>
    TDataType* pointer_cast(void* p);

    int32_t getNbOutputs() const noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const* inputs,
        int32_t                           nbInputs,
        nvinfer1::PluginTensorDesc const* outputs,
        int32_t                           nbOutputs
    ) const noexcept override;

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const*                inputs,
        void* const*                      outputs,
        void*                             workspace,
        cudaStream_t                      stream
    ) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(
        int32_t                           pos,
        nvinfer1::PluginTensorDesc const* inOut,
        int32_t                           nbInputs,
        int32_t                           nbOutputs
    ) noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::DataType getOutputDataType(
        int32_t                   index,
        nvinfer1::DataType const* inputTypes,
        int32_t                   nbInputs
    ) const noexcept override;

    void attachToContext(
        cudnnContext*            cudnn,
        cublasContext*           cublas,
        nvinfer1::IGpuAllocator* allocator
    ) noexcept override;

    void detachFromContext() noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in,
        int32_t                                  nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out,
        int32_t                                  nbOutputs
    ) noexcept override;

private:
    std::string mNamespace;

    // Q+1 (matrix dimension)
    int32_t mQp1{0};
    // epsilon regularizer
    float mEps{1e-6f};
    float mEta{1e-6f};

    // buffer for device arrays of pointers (for batched APIs)
    void* d_A_ptrs{nullptr};
    void* d_B_ptrs{nullptr};
    int*  d_info{nullptr};
    int*  d_pivots{nullptr}; // pivot indices (may be unused for some APIs but allocated)

    // cuBLAS handle
    cublasHandle_t mCublas;

    // cached shape parameters
    int32_t mBatch{0};
    int32_t mH{0};
    int32_t mW{0};
    int32_t mC{0};

    // plugin name/version
    static constexpr const char* PLUGIN_NAME{"LinearSolvePlugin"};
    static constexpr const char* PLUGIN_VERSION{"1"};
};

// Plugin Creator
class MOER_CUDA_API LinalgSolvePluginCreator : public nvinfer1::IPluginCreator {
public:
    LinalgSolvePluginCreator();

    ~LinalgSolvePluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2DynamicExt*
    createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2DynamicExt*
    deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection    mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string                        mNamespace;
};

#endif // LINALG_SOLVE_PLUGIN_H
