#ifndef GAUSSIAN_BLUR_PLUGIN_H
#define GAUSSIAN_BLUR_PLUGIN_H

#include "NvInferPlugin.h"
#include "NvInferRuntime.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "MoerCudaAPI.h"

namespace nvinfer1 { namespace plugin {

class MOER_CUDA_API GaussianBlurPlugin : public IPluginV2DynamicExt {
public:
    // Constructor for plugin creation
    GaussianBlurPlugin(float sigma);

    // Constructor for plugin cloning
    GaussianBlurPlugin(const void* data, size_t length);

    // Destructor
    ~GaussianBlurPlugin() override = default;

    // IPluginV2DynamicExt methods
    int getNbOutputs() const noexcept override;

    DimsExprs getOutputDimensions(
        int              outputIndex,
        const DimsExprs* inputs,
        int              nbInputs,
        IExprBuilder&    exprBuilder
    ) noexcept override;

    int  initialize() noexcept override;
    void terminate() noexcept override;

    size_t getWorkspaceSize(
        const PluginTensorDesc* inputs,
        int                     nbInputs,
        const PluginTensorDesc* outputs,
        int                     nbOutputs
    ) const noexcept override;

    int enqueue(
        const PluginTensorDesc* inputDesc,
        const PluginTensorDesc* outputDesc,
        const void* const*      inputs,
        void* const*            outputs,
        void*                   workspace,
        cudaStream_t            stream
    ) noexcept override;

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;

    void configurePlugin(
        const DynamicPluginTensorDesc* in,
        int                            nbInputs,
        const DynamicPluginTensorDesc* out,
        int                            nbOutputs
    ) noexcept override;

    bool supportsFormatCombination(
        int                     pos,
        const PluginTensorDesc* inOut,
        int                     nbInputs,
        int                     nbOutputs
    ) noexcept override;

    const char*          getPluginType() const noexcept override;
    const char*          getPluginVersion() const noexcept override;
    void                 destroy() noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    void                 setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char*          getPluginNamespace() const noexcept override;

    size_t getSerializationSize() const noexcept override;
    void   serialize(void* buffer) const noexcept override;

private:
    float       mSigma;
    std::string mNamespace;
};

class MOER_CUDA_API GaussianBlurPluginCreator : public IPluginCreator {
public:
    GaussianBlurPluginCreator();
    ~GaussianBlurPluginCreator() override = default;

    const char*                  getPluginName() const noexcept override;
    const char*                  getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt*
    deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void        setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection    mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string                     mNamespace;
};

}} // namespace nvinfer1::plugin

#endif // GAUSSIAN_BLUR_PLUGIN_H