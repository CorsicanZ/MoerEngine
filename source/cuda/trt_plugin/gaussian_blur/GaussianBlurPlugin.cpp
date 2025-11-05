#include "GaussianBlurPlugin.h"
#include "GaussianBlurKernels.h"
#include <cuda_fp16.h>
#include <cstring>
#include <cassert>
#include <iostream>
#include <vector>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace {
const char* GAUSSIAN_BLUR_PLUGIN_VERSION{"1"};
const char* GAUSSIAN_BLUR_PLUGIN_NAME{"GaussianBlurPlugin"};
} // namespace

// Plugin field collection
PluginFieldCollection GaussianBlurPluginCreator::mFC{};
std::vector<PluginField> GaussianBlurPluginCreator::mPluginAttributes;

// Helper functions for serialization/deserialization
template <typename T>
void writeToBuffer(char*& buffer, const T& val) {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
T readFromBuffer(const char*& buffer) {
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

// ============================================================================
// GaussianBlurPlugin Implementation
// ============================================================================

GaussianBlurPlugin::GaussianBlurPlugin(float sigma)
    : mSigma(sigma)
{
}

GaussianBlurPlugin::GaussianBlurPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    mSigma = readFromBuffer<float>(d);
}

int GaussianBlurPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs GaussianBlurPlugin::getOutputDimensions(
    int outputIndex,
    const DimsExprs* inputs,
    int nbInputs,
    IExprBuilder& exprBuilder
) noexcept
{
    // Output has same shape as input [N, C, H, W]
    assert(nbInputs == 1);
    assert(outputIndex == 0);
    return inputs[0];
}

int GaussianBlurPlugin::initialize() noexcept
{
    return 0;
}

void GaussianBlurPlugin::terminate() noexcept
{
}

size_t GaussianBlurPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs,
    int nbInputs,
    const PluginTensorDesc* outputs,
    int nbOutputs
) const noexcept
{
    // Need workspace for intermediate horizontal blur result
    // Size: N * C * H * W * sizeof(float) or sizeof(__half)
    const auto& dims = inputs[0].dims;
    size_t N = dims.d[0];
    size_t C = dims.d[1];
    size_t H = dims.d[2];
    size_t W = dims.d[3];

    size_t elemSize = (inputs[0].type == DataType::kFLOAT) ? sizeof(float) : sizeof(__half);
    return N * C * H * W * elemSize;
}

int GaussianBlurPlugin::enqueue(
    const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream
) noexcept
{
    // Get input dimensions [N, C, H, W]
    const int N = inputDesc[0].dims.d[0];
    const int C = inputDesc[0].dims.d[1];
    const int H = inputDesc[0].dims.d[2];
    const int W = inputDesc[0].dims.d[3];

    const void* input = inputs[0];
    void* output = outputs[0];

    // Check data type and launch appropriate kernel
    if (inputDesc[0].type == DataType::kFLOAT) {
        launchGaussianBlurFP32(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            static_cast<float*>(workspace),  // Pass workspace for intermediate result
            N, C, H, W,
            mSigma,
            stream
        );
    } else if (inputDesc[0].type == DataType::kHALF) {
        launchGaussianBlurFP16(
            static_cast<const __half*>(input),
            static_cast<__half*>(output),
            static_cast<__half*>(workspace),  // Pass workspace for intermediate result
            N, C, H, W,
            mSigma,
            stream
        );
    } else {
        std::cerr << "GaussianBlurPlugin: Unsupported data type" << std::endl;
        return -1;
    }

    return 0;
}

DataType GaussianBlurPlugin::getOutputDataType(
    int index,
    const DataType* inputTypes,
    int nbInputs
) const noexcept
{
    // Output has same data type as input
    assert(index == 0);
    return inputTypes[0];
}

void GaussianBlurPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in,
    int nbInputs,
    const DynamicPluginTensorDesc* out,
    int nbOutputs
) noexcept
{
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
}

bool GaussianBlurPlugin::supportsFormatCombination(
    int pos,
    const PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs
) noexcept
{
    assert(nbInputs == 1 && nbOutputs == 1);
    assert(pos < 2);

    // Support FP32 and FP16
    bool isValidType = (inOut[pos].type == DataType::kFLOAT ||
                       inOut[pos].type == DataType::kHALF);

    // Support linear format (NCHW)
    bool isValidFormat = (inOut[pos].format == TensorFormat::kLINEAR);

    // Input and output must have same type and format
    if (pos == 1) {
        isValidType = isValidType && (inOut[pos].type == inOut[0].type);
        isValidFormat = isValidFormat && (inOut[pos].format == inOut[0].format);
    }

    return isValidType && isValidFormat;
}

const char* GaussianBlurPlugin::getPluginType() const noexcept
{
    return GAUSSIAN_BLUR_PLUGIN_NAME;
}

const char* GaussianBlurPlugin::getPluginVersion() const noexcept
{
    return GAUSSIAN_BLUR_PLUGIN_VERSION;
}

void GaussianBlurPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* GaussianBlurPlugin::clone() const noexcept
{
    auto* plugin = new GaussianBlurPlugin(mSigma);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void GaussianBlurPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* GaussianBlurPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

size_t GaussianBlurPlugin::getSerializationSize() const noexcept
{
    return sizeof(mSigma);
}

void GaussianBlurPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    writeToBuffer(d, mSigma);
}

// ============================================================================
// GaussianBlurPluginCreator Implementation
// ============================================================================

GaussianBlurPluginCreator::GaussianBlurPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("sigma", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GaussianBlurPluginCreator::getPluginName() const noexcept
{
    return GAUSSIAN_BLUR_PLUGIN_NAME;
}

const char* GaussianBlurPluginCreator::getPluginVersion() const noexcept
{
    return GAUSSIAN_BLUR_PLUGIN_VERSION;
}

const PluginFieldCollection* GaussianBlurPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* GaussianBlurPluginCreator::createPlugin(
    const char* name,
    const PluginFieldCollection* fc
) noexcept
{
    float sigma = 0.8f; // Default value

    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attrName = fc->fields[i].name;
        if (!strcmp(attrName, "sigma")) {
            sigma = *static_cast<const float*>(fc->fields[i].data);
        }
    }

    auto* plugin = new GaussianBlurPlugin(sigma);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* GaussianBlurPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength
) noexcept
{
    auto* plugin = new GaussianBlurPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void GaussianBlurPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

const char* GaussianBlurPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// Register the plugin creator
REGISTER_TENSORRT_PLUGIN(GaussianBlurPluginCreator);