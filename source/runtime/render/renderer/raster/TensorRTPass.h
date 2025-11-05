/**
 * 此文件应该只有在宏 WITH_CUDA 被设置的情况下使用
 * 
 * 这个宏启用时，默认环境为Windows11+Vulkan；所以其他地方不再判断
 */
#pragma once

#if !defined(WITH_CUDA)
#error "This header requires WITH_CUDA=1"
#endif

#include "log/LogSystem.h"
#include "shader/ShaderPipeline.h"
#include "shaderheaders/shared/raster/post_process/ShaderParameters.h"

#include "CudaVulkanTools.h"
#include "RasterConfig.h"
#include "RasterResource.h"
#include "RasterTool.h"
#include "cuda_in_raster/cuda_in_raster.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>
#include <curand.h> // for rand
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

// plugin
#include "GaussianBlurPlugin.h"
#include "LinalgSolvePlugin.h"
#include <NvInferRuntime.h>

template<typename T>
class MoerPluginRegistrar {
public:
    MoerPluginRegistrar() {
        getPluginRegistry()->registerCreator(instance, "");
    }

private:
    //! Plugin instance.
    T instance{};
};
namespace nvinfer1::plugin {
static nvinfer1::PluginRegistrar<GaussianBlurPluginCreator> pluginRegistrarGaussianBlurPluginCreator{};
} // namespace nvinfer1::plugin
static nvinfer1::PluginRegistrar<LinalgSolvePluginCreator> pluginRegistrarLinalgSolvePluginCreator{};

using namespace nvinfer1;

namespace Moer::Render::Raster {

#define checkCudaErrors(val)     CudaVulkanTools::checkCudaErrorsInner((val), #val, __FILE__, __LINE__)
#define checkCusolverErrors(val) CudaVulkanTools::checkCusolverErrorsInner((val), #val, __FILE__, __LINE__)

struct TensorRTResource {

    CudaTexture ao;
    CudaTexture depth;
    CudaTexture color;
    CudaTexture motion;
    CudaTexture prev_ao;

    CudaSemaphore semaphore;

    // no RAII
    TensorRTResource(
        RasterContext& context,
        TextureRef     ao_tex,
        TextureRef     depth_tex,
        TextureRef     color_tex,
        TextureRef     motion_tex,
        TextureRef     prev_ao_tex
    ) :
        ao(context, ao_tex),
        depth(context, depth_tex),
        color(context, color_tex),
        motion(context, motion_tex),
        prev_ao(context, prev_ao_tex),
        semaphore(context) {

        ;
    };
    ~TensorRTResource() = default;

    TensorRTResource(const TensorRTResource&)            = delete;
    TensorRTResource& operator=(const TensorRTResource&) = delete;
};

class TensorRTLogger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            LOG_WARNING("[TRT] {}", msg);
    }
} gLogger;

/**
 * 这个类对 TRT Engine 进行封装
 * 
 * 这个类会根据传入的onnx_path，加载onnx文件，并把它编译成TRT Engine，并进行缓存
 * - 其中，缓存的代码是ai写的，不保证正确性；缓存的编译后结果位于目录 ./target/bin/Debug/resource/ai/tensorrt_cache/ 目录下
 * 
 * 这个类会对Engine的每个Input和Output，在gpu上建立对应大小的缓存
 * - 可以通过 device_mem_addr_map[channel_name] 来访问对应的gpu内存地址
 * 
 * FIXME: 这个struct不能使用UniquePtr，否则mimalloc会发出神秘错误，原因不明
 */
struct TensorRTEngine {

    std::unique_ptr<IBuilder>              builder;
    std::unique_ptr<INetworkDefinition>    network;
    std::unique_ptr<nvonnxparser::IParser> parser;
    std::unique_ptr<IBuilderConfig>        config;
    std::unique_ptr<ICudaEngine>           engine;
    std::unique_ptr<IExecutionContext>     context;

    UnorderedMap<std::string, __half*> device_mem_addr_map;
    UnorderedMap<std::string, size_t>  device_mem_size_map;

public:
    TensorRTEngine(const std::string& onnx_path, bool is_cache = false) {

        LOG_DEBUG("Prepare to load ONNX and build TensorRT Engine.");

        if (is_cache) {
            const std::string& cache_path = onnx_path;

            if (!LoadEngineFromCache(cache_path)) {
                LOG_INFO("Prebuilt TensorRT Engine is broken!");
            }
        } else {
            std::string cache_path = GetCachePath(onnx_path);

            // Try to load from cache first
            if (cache_path.empty() || !LoadEngineFromCache(cache_path)) { // writen by ai
                LOG_INFO("Cache not found or invalid, building engine from ONNX...");
                LoadEngineFromOnnx(onnx_path, cache_path);
            }
        }

        // Context
        context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
        if (!context) {
            LOG_ERROR("Context creation failed");
            return;
        }

        LOG_INFO("Created TensorRT IExecutionContext.");

        CreateBuffers(/* is_verbose */ true);
    }

    ~TensorRTEngine() {
        for (auto& kv : device_mem_addr_map) {
            if (kv.second != nullptr) {
                cudaFree(kv.second);
            }
        }
        device_mem_addr_map.clear();

        context.reset();
        engine.reset();

        config.reset();
        parser.reset();
        network.reset();
        builder.reset();
    }

    TensorRTEngine(const TensorRTEngine&)            = delete;
    TensorRTEngine& operator=(const TensorRTEngine&) = delete;

    // MARK: Load Random
    void LoadRandomValueToBuffers(TensorRTResource& res) {
        int nbIOTensors = engine->getNbIOTensors();

        for (int i = 0; i < nbIOTensors; ++i) {
            const char* name  = engine->getIOTensorName(i);
            Dims        shape = engine->getTensorShape(name);
            DataType    dtype = engine->getTensorDataType(name);

            if (strncmp(name, "in_", 3) == 0) {

                assert(shape.nbDims == 4);

                // 假设 shape 格式为 [N, C, H, W]
                int batch      = shape.d[0];
                int channels   = shape.d[1];
                int dst_height = shape.d[2]; // tensor 的目标高度
                int dst_width  = shape.d[3]; // tensor 的目标宽度

                size_t N = 1ULL * channels * dst_height * dst_width;
                dim3   blockSize(256);
                dim3   gridSize((N - 1) / blockSize.x / Moer::Cuda::RANDOMS_PER_THREAD);

                static uint64_t seed = 114514;

                Moer::Cuda::FillRandomHalf(
                    gridSize, blockSize, res.semaphore.stream_to_run, device_mem_addr_map[name], N, seed++
                );
            } else {
                checkCudaErrors(cudaMemset(device_mem_addr_map[name], 0, device_mem_size_map[name]));
            }
        }
    }

    // MARK: Load Zero
    void LoadZeroToBuffers() {
        int nbIOTensors = engine->getNbIOTensors();

        for (int i = 0; i < nbIOTensors; ++i) {
            const char* name  = engine->getIOTensorName(i);
            Dims        shape = engine->getTensorShape(name);
            DataType    dtype = engine->getTensorDataType(name);

            checkCudaErrors(cudaMemset(device_mem_addr_map[name], 0, device_mem_size_map[name]));
        }
    }

    // MARK: Engine1 Load
    void Engine1_LoadTexturesToBuffers(TensorRTResource& res, uint ao_only_idx, bool is_verbose) {

        int nbIOTensors = engine->getNbIOTensors();

        for (int i = 0; i < nbIOTensors; ++i) {
            const char* name  = engine->getIOTensorName(i);
            Dims        shape = engine->getTensorShape(name);
            DataType    dtype = engine->getTensorDataType(name);

            // 5.1 calculate array size

            // 假设 shape 格式为 [N, C, H, W]
            int batch      = shape.d[0];
            int channels   = shape.d[1];
            int dst_height = shape.d[2]; // tensor 的目标高度
            int dst_width  = shape.d[3]; // tensor 的目标宽度

            // CUDA kernel 执行配置 - 基于目标尺寸
            dim3 blockSize(16, 16);
            dim3 gridSize(
                (dst_width + blockSize.x - 1) / blockSize.x, (dst_height + blockSize.y - 1) / blockSize.y
            );

            /**
             * 默认所有数据从RGBA开始填，即 depth占R；motion vector占RG
             */
            auto copy_to_buf = [&](CudaTexture& src_tex, int channels, __half* d_target) {
                CudaTexture::EFormatElementType type       = src_tex.GetElementType();
                size_t                          type_count = src_tex.GetElementTypeCount();

                if (type == CudaTexture::EFormatElementType::UCHAR && type_count == 4) {
                    Moer::Cuda::CopySurfaceToBuffer_Resize_NCHW_Half_Uchar4(
                        gridSize,
                        blockSize,
                        res.semaphore.stream_to_run,
                        // 0, // no stream object
                        src_tex.GetSurfaceObjectList(),
                        d_target,
                        src_tex.width,
                        src_tex.height,
                        dst_width,
                        dst_height,
                        channels
                    );
                } else if (type == CudaTexture::EFormatElementType::UCHAR && type_count == 1) {
                    Moer::Cuda::CopySurfaceToBuffer_Resize_NCHW_Half_Uchar1(
                        gridSize,
                        blockSize,
                        res.semaphore.stream_to_run,
                        // 0, // no stream object
                        src_tex.GetSurfaceObjectList(),
                        d_target,
                        src_tex.width,
                        src_tex.height,
                        dst_width,
                        dst_height,
                        channels
                    );
                } else if (type == CudaTexture::EFormatElementType::FLOAT && type_count == 4) {
                    Moer::Cuda::CopySurfaceToBuffer_Resize_NCHW_Half_Float4(
                        gridSize,
                        blockSize,
                        res.semaphore.stream_to_run,
                        // 0, // no stream object
                        src_tex.GetSurfaceObjectList(),
                        d_target,
                        src_tex.width,
                        src_tex.height,
                        dst_width,
                        dst_height,
                        channels
                    );
                } else if (type == CudaTexture::EFormatElementType::FLOAT && type_count == 1) {
                    Moer::Cuda::CopySurfaceToBuffer_Resize_NCHW_Half_Float1(
                        gridSize,
                        blockSize,
                        res.semaphore.stream_to_run,
                        // 0, // no stream object
                        src_tex.GetSurfaceObjectList(),
                        d_target,
                        src_tex.width,
                        src_tex.height,
                        dst_width,
                        dst_height,
                        channels
                    );
                } else if (type == CudaTexture::EFormatElementType::HALF && type_count == 2) {
                    Moer::Cuda::CopySurfaceToBuffer_Resize_NCHW_Half_Half2(
                        gridSize,
                        blockSize,
                        res.semaphore.stream_to_run,
                        // 0, // no stream object
                        src_tex.GetSurfaceObjectList(),
                        d_target,
                        src_tex.width,
                        src_tex.height,
                        dst_width,
                        dst_height,
                        channels
                    );
                } else {
                    assert(false);
                }

                if (is_verbose) {
                    LOG_DEBUG(
                        "tex {}: size from ({}, {}) to ({}, {}); channels = {}; type = {}{}",
                        name,
                        src_tex.width,
                        src_tex.height,
                        dst_width,
                        dst_height,
                        channels,
                        (type == CudaTexture::EFormatElementType::FLOAT ?
                             "float" :
                             (type == CudaTexture::EFormatElementType::HALF ? "half" : "uchar")),
                        type_count
                    );
                }
            };

            if (std::strcmp(name, "in_ao") == 0) {
                copy_to_buf((ao_only_idx ? res.prev_ao : res.ao), 1, device_mem_addr_map[name]);

            } else if (std::strcmp(name, "in_depth") == 0) {
                copy_to_buf(res.depth, 1, device_mem_addr_map[name]);

            } else if (std::strcmp(name, "in_color") == 0) {
                copy_to_buf(res.color, 3, device_mem_addr_map[name]);

            } else if (std::strcmp(name, "in_motion") == 0) {
                copy_to_buf(res.motion, 2, device_mem_addr_map[name]);

            } else if (std::strcmp(name, "in_prev_ao") == 0) {
                copy_to_buf((ao_only_idx ? res.ao : res.prev_ao), 1, device_mem_addr_map[name]);

            } else if (std::strcmp(name, "in_prev_embed") == 0) {

                static bool isFirstTime = true;
                if (isFirstTime) {
                    isFirstTime = false;

                    // 第一次执行，置0
                    checkCudaErrors(cudaMemset(device_mem_addr_map[name], 0, device_mem_size_map[name]));

                } else {

                    // 第N次执行，设置为 out_embed
                    checkCudaErrors(cudaMemcpy(
                        device_mem_addr_map[name],
                        device_mem_addr_map["out_embed"],
                        device_mem_size_map[name],
                        cudaMemcpyDeviceToDevice
                    ));
                }

            } else {
                // random value should be ok
                // checkCudaErrors(cudaMemset(device_mem_addr_map[name], 0, device_mem_size_map[name]));
            }
        }
    }

    // MARK: Engine2 Load
    void Engine2_LoadEngine1OutputToBuffers(
        const cudaStream_t& stream_to_run,
        TensorRTEngine&     engine1,
        cusolverDnHandle_t  cusolver
    ) {

        int nbIOTensors = engine->getNbIOTensors();

        for (int i = 0; i < nbIOTensors; ++i) {
            const char* name  = engine->getIOTensorName(i);
            Dims        shape = engine->getTensorShape(name);
            DataType    dtype = engine->getTensorDataType(name);

            // TODO: 优化空间，考虑0开销拷贝
            auto copy_buf_between_engine = [&](const char* engine1_name, const char* engine2_name) {
                checkCudaErrors(cudaMemcpy(
                    device_mem_addr_map[engine2_name],
                    engine1.device_mem_addr_map[engine1_name],
                    device_mem_size_map[engine2_name],
                    cudaMemcpyDeviceToDevice
                ));

                assert(engine1.device_mem_size_map[engine1_name] == device_mem_size_map[engine2_name]);
            };

            if (std::strcmp(name, "in_X_model") == 0) {
                copy_buf_between_engine("out_X_model", name);

            } else if (std::strcmp(name, "in_coeffs_batch") == 0) {

                Moer::Cuda::SolveBatchedFXP16(
                    stream_to_run,
                    cusolver,
                    shape.d[0],
                    shape.d[1],
                    shape.d[2],
                    engine1.device_mem_addr_map["out_XTX_batch"],
                    engine1.device_mem_addr_map["out_XTY_batch"],
                    device_mem_addr_map[name],
                    1e-3
                );

            } else if (std::strcmp(name, "in_upscale_kernel") == 0) {
                copy_buf_between_engine("out_upscale_kernel", name);

            } else if (std::strcmp(name, "in_color") == 0) {
                copy_buf_between_engine("out_color", name);

            } else if (std::strcmp(name, "in_prev_ao") == 0) {
                copy_buf_between_engine("out_prev_ao", name);

            } else {
                // random value should be ok
                // checkCudaErrors(cudaMemset(device_mem_addr_map[name], 0, device_mem_size_map[name]));
            }
        }
    }

    // MARK: EngineNeedsPlugin Load
    /**
     * Engine Unnamed Network 0 has 16 I/O tensors:
     * Tensor[0 ] (Input ) name = ao; element_size = 2; shape = (1, 1, 540, 540, );
     * Tensor[1 ] (Input ) name = depth; element_size = 2; shape = (1, 1, 540, 540, );
     * Tensor[2 ] (Input ) name = color; element_size = 2; shape = (1, 3, 540, 540, ); 
     * Tensor[3 ] (Input ) name = motion; element_size = 2; shape = (1, 2, 540, 540, );
     * Tensor[4 ] (Input ) name = temporal_ao; element_size = 2; shape = (1, 1, 540, 540, );  
     * Tensor[5 ] (Input ) name = temporal_embed; element_size = 2; shape = (1, 32, 540, 540, );
     * 
     * Tensor[6 ] (Input ) name = LinalgSolve_XTX; element_size = 2; shape = (2, 540, 960, 9, );
     * Tensor[7 ] (Input ) name = LinalgSolve_XTY; element_size = 2; shape = (2, 540, 960, 9, );
     * Tensor[8 ] (Input ) name = GaussianBlur_Input; element_size = 2; shape = (2, 4, 68, 120, );    
     * 
     * Tensor[9 ] (Output) name = final_output; element_size = 2; shape = (1, 3, 1080, 1080, );
     * Tensor[10] (Output) name = denoised; element_size = 2; shape = (1, 1, 540, 540, );    
     * Tensor[11] (Output) name = new_temporal_ao; element_size = 2; shape = (1, 1, 540, 540, );     
     * Tensor[12] (Output) name = new_temporal_embed; element_size = 2; shape = (1, 32, 540, 540, );
     * Tensor[13] (Output) name = grid; element_size = 2; shape = (1, 540, 540, 2, ); 
     * 
     * Tensor[14] (Output) name = LinalgSolve_Output; element_size = 2; shape = (2, 540, 960, 9, );
     * Tensor[15] (Output) name = GaussianBlur_Output; element_size = 2; shape = (2, 4, 68, 120, );  
     */
    void EngineNeedsPlugin_LoadTexturesToBuffers(TensorRTResource& res, uint ao_only_idx, bool is_verbose) {
        int nbIOTensors = engine->getNbIOTensors();

        for (int i = 0; i < nbIOTensors; ++i) {
            const char* name  = engine->getIOTensorName(i);
            Dims        shape = engine->getTensorShape(name);
            DataType    dtype = engine->getTensorDataType(name);

            // 5.1 calculate array size

            // 假设 shape 格式为 [N, C, H, W]
            int batch      = shape.d[0];
            int channels   = shape.d[1];
            int dst_height = shape.d[2]; // tensor 的目标高度
            int dst_width  = shape.d[3]; // tensor 的目标宽度

            // CUDA kernel 执行配置 - 基于目标尺寸
            dim3 blockSize(16, 16);
            dim3 gridSize(
                (dst_width + blockSize.x - 1) / blockSize.x, (dst_height + blockSize.y - 1) / blockSize.y
            );

            /**
             * 默认所有数据从RGBA开始填，即 depth占R；motion vector占RG
             */
            auto copy_to_buf = [&](CudaTexture& src_tex, int channels, __half* d_target) {
                CudaTexture::EFormatElementType type       = src_tex.GetElementType();
                size_t                          type_count = src_tex.GetElementTypeCount();

                if (type == CudaTexture::EFormatElementType::UCHAR && type_count == 4) {
                    Moer::Cuda::CopySurfaceToBuffer_Resize_NCHW_Half_Uchar4(
                        gridSize,
                        blockSize,
                        res.semaphore.stream_to_run,
                        // 0, // no stream object
                        src_tex.GetSurfaceObjectList(),
                        d_target,
                        src_tex.width,
                        src_tex.height,
                        dst_width,
                        dst_height,
                        channels
                    );
                } else if (type == CudaTexture::EFormatElementType::UCHAR && type_count == 1) {
                    Moer::Cuda::CopySurfaceToBuffer_Resize_NCHW_Half_Uchar1(
                        gridSize,
                        blockSize,
                        res.semaphore.stream_to_run,
                        // 0, // no stream object
                        src_tex.GetSurfaceObjectList(),
                        d_target,
                        src_tex.width,
                        src_tex.height,
                        dst_width,
                        dst_height,
                        channels
                    );
                } else if (type == CudaTexture::EFormatElementType::FLOAT && type_count == 4) {
                    Moer::Cuda::CopySurfaceToBuffer_Resize_NCHW_Half_Float4(
                        gridSize,
                        blockSize,
                        res.semaphore.stream_to_run,
                        // 0, // no stream object
                        src_tex.GetSurfaceObjectList(),
                        d_target,
                        src_tex.width,
                        src_tex.height,
                        dst_width,
                        dst_height,
                        channels
                    );
                } else if (type == CudaTexture::EFormatElementType::FLOAT && type_count == 1) {
                    Moer::Cuda::CopySurfaceToBuffer_Resize_NCHW_Half_Float1(
                        gridSize,
                        blockSize,
                        res.semaphore.stream_to_run,
                        // 0, // no stream object
                        src_tex.GetSurfaceObjectList(),
                        d_target,
                        src_tex.width,
                        src_tex.height,
                        dst_width,
                        dst_height,
                        channels
                    );
                } else if (type == CudaTexture::EFormatElementType::HALF && type_count == 2) {
                    Moer::Cuda::CopySurfaceToBuffer_Resize_NCHW_Half_Half2(
                        gridSize,
                        blockSize,
                        res.semaphore.stream_to_run,
                        // 0, // no stream object
                        src_tex.GetSurfaceObjectList(),
                        d_target,
                        src_tex.width,
                        src_tex.height,
                        dst_width,
                        dst_height,
                        channels
                    );
                } else {
                    assert(false);
                }

                if (is_verbose) {
                    LOG_DEBUG(
                        "tex {}: size from ({}, {}) to ({}, {}); channels = {}; type = {}{}",
                        name,
                        src_tex.width,
                        src_tex.height,
                        dst_width,
                        dst_height,
                        channels,
                        (type == CudaTexture::EFormatElementType::FLOAT ?
                             "float" :
                             (type == CudaTexture::EFormatElementType::HALF ? "half" : "uchar")),
                        type_count
                    );
                }
            };

            if (std::strcmp(name, "ao") == 0) {
                copy_to_buf((ao_only_idx ? res.prev_ao : res.ao), 1, device_mem_addr_map[name]);

            } else if (std::strcmp(name, "depth") == 0) {
                copy_to_buf(res.depth, 1, device_mem_addr_map[name]);

            } else if (std::strcmp(name, "color") == 0) {
                copy_to_buf(res.color, 3, device_mem_addr_map[name]);

            } else if (std::strcmp(name, "motion") == 0) {
                copy_to_buf(res.motion, 2, device_mem_addr_map[name]);

            } else if (std::strcmp(name, "temporal_ao") == 0) {
                copy_to_buf((ao_only_idx ? res.ao : res.prev_ao), 1, device_mem_addr_map[name]);

            } else if (std::strcmp(name, "temporal_embed") == 0) {

                static bool isFirstTime = true;
                if (isFirstTime) {
                    isFirstTime = false;

                    // 第一次执行，置0
                    checkCudaErrors(cudaMemset(device_mem_addr_map[name], 0, device_mem_size_map[name]));

                } else {

                    // 第N次执行，设置为 out_embed
                    checkCudaErrors(cudaMemcpy(
                        device_mem_addr_map[name],
                        device_mem_addr_map["new_temporal_embed"],
                        device_mem_size_map[name],
                        cudaMemcpyDeviceToDevice
                    ));
                }

            } else {
                // random value should be ok
                // checkCudaErrors(cudaMemset(device_mem_addr_map[name], 0, device_mem_size_map[name]));
            }
        }
    }

    void Run(const cudaStream_t& stream_to_run) {
        bool is_success = context->enqueueV3(stream_to_run);
        if (!is_success) {
            LOG_ERROR("TRT Pass: enqueueV3 failed.");
        }
    }

private:
    std::string GetCachePath(const std::string& onnx_file_path) { // writen by ai
        auto cache_dir = ConfigManager::GetInstance().GetEditorResourcePath() / "ai" / "tensorrt_cache";
        std::filesystem::create_directories(cache_dir);

        // Generate hash from ONNX file content
        std::ifstream file(onnx_file_path, std::ios::binary);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open ONNX file for hashing: {}", onnx_file_path);
            return "";
        }

        std::hash<std::string> hasher;
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        size_t      hash_value = hasher(content);

        return (cache_dir / (std::to_string(hash_value) + ".trt")).string();
    }

    bool LoadEngineFromCache(const std::string& cache_file_path) { // writen by ai
        std::ifstream file(cache_file_path, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
        if (!runtime) {
            LOG_ERROR("Failed to create TensorRT runtime for cache loading");
            return false;
        }

        engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), size));
        if (!engine) {
            LOG_ERROR("Failed to deserialize cached engine");
            return false;
        }

        LOG_DEBUG("Successfully loaded TensorRT engine from cache: {}", cache_file_path);
        return true;
    }

    bool SaveEngineToCache(const std::string& cache_file_path) { // writen by ai
        if (!engine) {
            LOG_ERROR("No engine to save to cache");
            return false;
        }

        auto serialized_engine = std::unique_ptr<IHostMemory>(engine->serialize());
        if (!serialized_engine) {
            LOG_ERROR("Failed to serialize engine");
            return false;
        }

        std::ofstream file(cache_file_path, std::ios::binary);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open cache file for writing: {}", cache_file_path);
            return false;
        }

        file.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
        file.close();

        LOG_INFO("Successfully saved TensorRT engine to cache: {}", cache_file_path);
        return true;
    }

    // MARK: LoadEngine ONNX
    void LoadEngineFromOnnx(const std::string& onnx_path, const std::string& cache_path) {

        initLibNvInferPlugins(&gLogger, "");

        {
            IPluginRegistry*         registry = getPluginRegistry();
            IPluginCreatorInterface* creator  = registry->getCreator("LinalgSolvePlugin", "1");
            LOG_DEBUG("LinalgSolvePlugin.1 - creator {}", (creator == nullptr ? "is nullptr" : "found"));
        }
        {
            IPluginRegistry*         registry = getPluginRegistry();
            IPluginCreatorInterface* creator  = registry->getCreator("GaussianBlurPlugin", "1");
            LOG_DEBUG("GaussianBlurPlugin.1 - creator {}", (creator == nullptr ? "is nullptr" : "found"));
        }

        // 1. Builder
        {
            builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));

            // 检查设备是否支持 FP16
            if (!builder->platformHasFastFp16()) {
                LOG_WARNING("Platform does not have fast FP16 support");
            }

            // 设置多线程构建
            int32_t numThreads = std::thread::hardware_concurrency();
            if (!builder->setMaxThreads(numThreads)) {
                LOG_WARNING("Failed to set max threads to {}, using default", numThreads);
            } else {
                LOG_DEBUG("Builder configured to use {} threads", builder->getMaxThreads());
            }
        }
        // 1.1 Network
        {
            // // 创建强类型网络（推荐方式）
            // NetworkDefinitionCreationFlags flags =
            //     1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
            // network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(flags));

            // 普通网络
            network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(0));
        }
        // 1.2 Parser
        {
            parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));

            if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
                LOG_ERROR("ONNX parse failed. Please check the ONNX file path!");
                return;
            }
        }
        // 1.2.5 Print All Layers
        if (false) {
            // Print per-layer / per-tensor data types for debugging
            auto dtypeToString = [](DataType t) {
                switch (t) {
                    case DataType::kFLOAT:
                        return "kFLOAT";
                    case DataType::kHALF:
                        return "kHALF";
                    case DataType::kINT8:
                        return "kINT8";
                    case DataType::kINT32:
                        return "kINT32";
                    case DataType::kINT64:
                        return "kINT64";
                    default:
                        LOG_ERROR("Unknown DataType {}", static_cast<uint>(t));
                        return "UNKNOWN";
                }
            };

            LOG_DEBUG(
                "Network summary: layers={} inputs={} outputs={}",
                network->getNbLayers(),
                network->getNbInputs(),
                network->getNbOutputs()
            );

            // Layers: list inputs/outputs and their dtypes
            for (int li = 0; li < network->getNbLayers(); ++li) {
                ILayer*     layer = network->getLayer(li);
                const char* lname = layer->getName() ? layer->getName() : "<noname>";
                // LOG_DEBUG(
                //     "Layer[{}] name='{}' nbInputs={} nbOutputs={}",
                //     li,
                //     lname,
                //     layer->getNbInputs(),
                //     layer->getNbOutputs()
                // );

                for (int in = 0; in < layer->getNbInputs(); ++in) {
                    ITensor* t = layer->getInput(in);
                    if (t) {
                        const char* tname = t->getName() ? t->getName() : "<noname>";
                        LOG_DEBUG("  input[{}] name='{}' dtype={}", in, tname, dtypeToString(t->getType()));
                    }
                }
                for (int out = 0; out < layer->getNbOutputs(); ++out) {
                    ITensor* t = layer->getOutput(out);
                    if (t) {
                        const char* tname = t->getName() ? t->getName() : "<noname>";
                        LOG_DEBUG("  output[{}] name='{}' dtype={}", out, tname, dtypeToString(t->getType()));
                    }
                }
            }

            // Network-level inputs/outputs
            for (int i = 0; i < network->getNbInputs(); ++i) {
                ITensor*    t     = network->getInput(i);
                const char* tname = t->getName() ? t->getName() : "<noname>";
                LOG_DEBUG("NetworkInput[{}] name='{}' dtype={}", i, tname, dtypeToString(t->getType()));
            }
            for (int i = 0; i < network->getNbOutputs(); ++i) {
                ITensor*    t     = network->getOutput(i);
                const char* tname = t->getName() ? t->getName() : "<noname>";
                LOG_DEBUG("NetworkOutput[{}] name='{}' dtype={}", i, tname, dtypeToString(t->getType()));
            }
        }
        // 1.3 Convert to FP16
        {
            // 设置输入精度
            for (int i = 0; i < network->getNbInputs(); i++) {
                network->getInput(i)->setType(DataType::kHALF);
            }

            // 设置输出精度
            for (int i = 0; i < network->getNbOutputs(); i++) {
                network->getOutput(i)->setType(DataType::kHALF);
            }

            LOG_DEBUG(
                "Set all network tensors to FP16. Layers/Inputs/Outputs: {}/{}/{}",
                network->getNbLayers(),
                network->getNbInputs(),
                network->getNbOutputs()
            );
        }

        // MARK: LinalgSolvePlugin
        {
            nvinfer1::IPluginRegistry* registry = getPluginRegistry();
            auto                       creator  = registry->getPluginCreator("LinalgSolvePlugin", "1");
            if (!creator) {
                LOG_ERROR("Cannot find LinalgSolvePluginCreator");
                return;
            }
            LOG_DEBUG("Found plugin creator: {}", creator->getPluginName());

            int   B = 2, H = 540, W = 960, Q_plus_1 = 3, C = 3;
            float epsilon = 0.01f;
            float eta     = 0.001f;

            // 2. 设置 Plugin 参数
            int         q_val = Q_plus_1;
            PluginField fields[3];
            fields[0].name   = "q_plus_one";
            fields[0].data   = &q_val;
            fields[0].type   = PluginFieldType::kINT32;
            fields[0].length = 1;

            fields[1].name   = "epsilon";
            fields[1].data   = &epsilon;
            fields[1].type   = PluginFieldType::kFLOAT32;
            fields[1].length = 1;

            fields[2].name   = "eta";
            fields[2].data   = &eta;
            fields[2].type   = PluginFieldType::kFLOAT32;
            fields[2].length = 1;

            PluginFieldCollection fc;
            fc.nbFields = 3;
            fc.fields   = fields;

            IPluginV2* plugin = creator->createPlugin("LinalgSolve", &fc);
            LOG_DEBUG("Created plugin instance: {}", creator->getPluginName());

            // 4. 输入定义
            Dims XTX_shape{4, {B, H, W, Q_plus_1 * Q_plus_1}};
            Dims XTY_shape{4, {B, H, W, Q_plus_1 * C}};

            // ITensor* input_XTX = network->addInput("LinalgSolve_XTX", DataType::kFLOAT, XTX_shape);
            // ITensor* input_XTY = network->addInput("LinalgSolve_XTY", DataType::kFLOAT, XTY_shape);
            ITensor* input_XTX = network->addInput("LinalgSolve_XTX", DataType::kHALF, XTX_shape);
            ITensor* input_XTY = network->addInput("LinalgSolve_XTY", DataType::kHALF, XTY_shape);

            // 5. 插入 Plugin
            ITensor* plugin_inputs[] = {input_XTX, input_XTY};
            ILayer*  linalg_layer    = network->addPluginV2(plugin_inputs, 2, *plugin);

            // 6. 标记输出
            auto* output_layer = linalg_layer->getOutput(0);
            output_layer->setName("LinalgSolve_Output");
            output_layer->setType(DataType::kHALF);
            network->markOutput(*output_layer);
        }
        // MARK: GaussianBlurPlugin
        {
            nvinfer1::IPluginRegistry* registry = getPluginRegistry();
            auto                       creator  = registry->getPluginCreator("GaussianBlurPlugin", "1");
            if (!creator) {
                LOG_ERROR("Cannot find GaussianBlurPluginCreator");
                return;
            }
            LOG_DEBUG("Found plugin creator: {}", creator->getPluginName());

            int   B = 2, C = 4, H = 68, W = 120;
            float sigma = 1.5f;

            // 2. 设置 Plugin 参数 (sigma)
            PluginField fields[1];
            fields[0].name   = "sigma";
            fields[0].data   = &sigma;
            fields[0].type   = PluginFieldType::kFLOAT32;
            fields[0].length = 1;

            PluginFieldCollection fc;
            fc.nbFields = 1;
            fc.fields   = fields;

            IPluginV2* plugin = creator->createPlugin("GaussianBlur", &fc);
            LOG_DEBUG("Created plugin instance: {}. Sigma: {}", creator->getPluginName(), sigma);

            // 4. 定义网络输入 (NCHW)
            Dims input_dims{4, {B, C, H, W}};
            // ITensor* input_tensor = network->addInput("GaussianBlur_Input", DataType::kFLOAT, input_dims);
            ITensor* input_tensor = network->addInput("GaussianBlur_Input", DataType::kHALF, input_dims);

            // 5. 插入 Plugin
            ITensor* plugin_inputs[] = {input_tensor};
            ILayer*  gaussian_layer  = network->addPluginV2(plugin_inputs, 1, *plugin);

            // 6. 标记输出
            auto* output_layer = gaussian_layer->getOutput(0);
            output_layer->setName("GaussianBlur_Output");
            output_layer->setType(DataType::kHALF);
            network->markOutput(*output_layer);
        }

        // 2. Config
        config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB

        // 可选的构建优化
        config->setBuilderOptimizationLevel(3); // 默认级别

        // // 强制fp16
        // // trt10.12废弃的方法，但是简单
        // {
        //     config->setFlag(BuilderFlag::kFP16);
        //     config->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
        // }

        // cublas，设置 tactic source
        {
            config->setTacticSources(
                config->getTacticSources() | (1 << static_cast<int>(TacticSource::kCUBLAS))
            );
        }

        auto profile = builder->createOptimizationProfile();
        config->addOptimizationProfile(profile);

        // 3. Engine
        LOG_DEBUG("Started to Build TensorRT ICudaEngine. Needs to wait several minutes.");

        engine = std::unique_ptr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
        if (!engine) {
            LOG_ERROR("Engine build failed");
            return;
        }

        LOG_DEBUG("Created TensorRT ICudaEngine.");

        // Save newly built engine to cache
        if (!cache_path.empty()) {
            SaveEngineToCache(cache_path);
        }
    }

    // MARK: CreateBuffers
    void CreateBuffers(bool is_verbose) {

        auto get_element_size = [](const DataType& dtype) {
            switch (dtype) {
                case DataType::kFLOAT:
                    return sizeof(float);
                case DataType::kHALF:
                    return sizeof(__half);
                case DataType::kINT8:
                    return sizeof(int8_t);
                case DataType::kINT32:
                    return sizeof(int32_t);
                default:
                    assert(false);
            };
        };

        auto get_element_count = [](const Dims& shape) {
            size_t sum = 1;
            for (int i = 0; i < shape.nbDims; i++) {
                sum *= shape.d[i];
            }
            return sum;
        };

        int nbIOTensors = engine->getNbIOTensors();

        std::ostringstream output_stream;
        output_stream << "\nEngine " << engine->getName() << " has " << nbIOTensors << " I/O tensors:\n";

        auto create_buf_for_tensor = [&](const char* name, const char* prefix, int i) {
            Dims     shape = engine->getTensorShape(name);
            DataType dtype = engine->getTensorDataType(name);

            size_t element_size  = get_element_size(dtype);
            size_t element_count = get_element_count(shape);
            size_t total_bytes   = element_count * element_size;

            assert(element_size == sizeof(__half));

            // malloc
            __half* d_memory = nullptr; // device memory
            checkCudaErrors(cudaMalloc(&d_memory, total_bytes));

            // set 0 for all bytes
            checkCudaErrors(cudaMemset(d_memory, 0, total_bytes));

            device_mem_addr_map[name] = d_memory;
            device_mem_size_map[name] = total_bytes;

            // 5.3 bind

            context->setTensorAddress(name, d_memory);

            // 5.4 output

            output_stream << "Tensor[" << i << "] (" << prefix << ") name = " << name
                          << "; element_size = " << element_size << "; shape = (";
            for (int i = 0; i < shape.nbDims; i++)
                output_stream << shape.d[i] << ", ";
            output_stream << ");";

            // output_stream << "\tbuffer length = " << element_count << ";\tbuffer size = " << total_bytes / 1024
            //               << "KB";
            output_stream << "\n";
        };

        std::vector<const char*> input_tensors;
        std::vector<const char*> output_tensors;

        for (int i = 0; i < nbIOTensors; ++i) {
            const char* name = engine->getIOTensorName(i);

            nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                input_tensors.push_back(name);
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                output_tensors.push_back(name);
            } else {
                assert(false);
            }
        }

        for (int i = 0; i < input_tensors.size(); ++i) {
            create_buf_for_tensor(input_tensors[i], "Input ", i);
        }
        for (int i = 0; i < output_tensors.size(); ++i) {
            create_buf_for_tensor(output_tensors[i], "Output", input_tensors.size() + i);
        }

        if (is_verbose) {
            LOG_DEBUG("TensorRT Buffers Info: {}", output_stream.str());
        }
    }
};

/**
 * MARK: TensorRT Pass
 */
class TensorRTPass {

private:
    RasterContext& context;

    UniquePtr<TensorRTResource> res;
    UniquePtr<TensorRTEngine>   engine_needs_plugin;

    cusolverDnHandle_t cusolver = nullptr;

public:
    TensorRTPass(
        RasterContext& _context,
        TextureRef     ao_tex,
        TextureRef     depth_tex,
        TextureRef     color_tex,
        TextureRef     motion_tex,
        TextureRef     prev_ao_tex
    ) :
        context(_context) {

        // cusolver
        checkCusolverErrors(cusolverDnCreate(&cusolver));

        res = MakeUnique<TensorRTResource>(context, ao_tex, depth_tex, color_tex, motion_tex, prev_ao_tex);

        engine_needs_plugin = MakeUnique<TensorRTEngine>(
            (ConfigManager::GetInstance().GetEditorResourcePath() / "ai" / "onnx_models" / "flnr41103.onnx")
                .string()
        );
    }
    ~TensorRTPass() {
        engine_needs_plugin.reset();
        res.reset();
        cusolverDnDestroy(cusolver);
    }

    TensorRTPass(const TensorRTPass&)            = delete;
    TensorRTPass& operator=(const TensorRTPass&) = delete;

    void RecreateResource(
        RasterContext& context,
        TextureRef     ao_tex,
        TextureRef     depth_tex,
        TextureRef     color_tex,
        TextureRef     motion_tex,
        TextureRef     prev_ao_tex
    ) {
        res.reset();
    }

    uint Process(RasterContext& context, const RasterConfig& ui_config, uint input_image, uint ao_only_idx) {
        assert(ui_config.ai_is_cuda_enabled);

        // signal

        res->semaphore.Signal();

        // cuda

        // TODO: 优化一下 cudaStreamSynchronize(res->semaphore.stream_to_run)
        auto sync = [&]() {
            checkCudaErrors(cudaStreamSynchronize(res->semaphore.stream_to_run));
        };

        // engine1->LoadRandomValueToBuffers(*res);
        // engine1->LoadZeroToBuffers();
        engine_needs_plugin->EngineNeedsPlugin_LoadTexturesToBuffers(*res, ao_only_idx, false);
        sync();

        engine_needs_plugin->Run(res->semaphore.stream_to_run);
        sync();

        // CheckBuf();

        VisualizeFeature(
            *engine_needs_plugin,
            res->color,
            ui_config.ai_trt_visualize_buffer.c_str(),
            res->semaphore.stream_to_run,
            ui_config.ai_cuda_pass_debug_param
        );

        sync();

        // wait

        res->semaphore.Wait();

        // return

        return input_image;
    }

private:
    // MARK: Visualize
    void VisualizeFeature(
        TensorRTEngine&     engine,
        CudaTexture&        color,
        const char*         name,
        const cudaStream_t& stream_to_run,
        float               debug_param
    ) {

        const static uint64 TILE = 16;

        dim3 threadsPerBlock(TILE, TILE);
        dim3 blocksPerGrid(
            (color.width - 1) / threadsPerBlock.x + 1, (color.height - 1) / threadsPerBlock.y + 1
        );

        if (engine.engine->getTensorShape(name).nbDims != 4) {
            LOG_WARNING("{}.shape != 4", name);
            return;
        }
        if (engine.device_mem_addr_map.contains(name) == false) {
            LOG_WARNING("Cannot find this buffer: {}", name);
            return;
        }

        Moer::Cuda::VisualizeFeatureBuf(
            blocksPerGrid,
            threadsPerBlock,
            stream_to_run,
            color.GetSurfaceObjectList(),
            (__half*)engine.device_mem_addr_map[name],
            engine.engine->getTensorShape(name).d[3], // width
            engine.engine->getTensorShape(name).d[2], // height
            engine.engine->getTensorShape(name).d[1], // channels
            color.width,
            color.height,
            debug_param
        );
    }

    void CheckBuf() {
        // 检查 final_output buffer 的值：拷回 host 并统计 min/max/是否全部为 0
        auto check_buf = [&](TensorRTEngine& engine, const char* name) {
            auto it = engine.device_mem_addr_map.find(name);
            if (it == engine.device_mem_addr_map.end()) {
                return;
            }

            void*    d_final_output = it->second;
            Dims     shape          = engine.engine->getTensorShape(name);
            DataType dtype          = engine.engine->getTensorDataType(name);

            // Check if all values in final_output buffer are the same
            size_t element_count = 1;
            for (int i = 0; i < shape.nbDims; i++) {
                element_count *= shape.d[i];
            }

            std::vector<__half> h_final_output(element_count);
            checkCudaErrors(cudaMemcpy(
                h_final_output.data(), d_final_output, element_count * sizeof(__half), cudaMemcpyDeviceToHost
            ));

            bool   all_same  = true;
            __half first_val = h_final_output[0];
            __half min_val   = h_final_output[0];
            __half max_val   = h_final_output[0];
            for (size_t i = 1; i < element_count; i++) {
                if (std::abs(float(h_final_output[i] - first_val)) > 0.0001) {
                    all_same = false;
                }
                min_val = std::min(min_val, h_final_output[i]);
                max_val = std::max(max_val, h_final_output[i]);
            }

            LOG_DEBUG(
                "\tcheck {}: all values same = {}, first value = {}, min = {}, max = {}",
                name,
                all_same,
                static_cast<float>(first_val),
                static_cast<float>(min_val),
                static_cast<float>(max_val)
            );
        };

        LOG_DEBUG("");
        LOG_DEBUG("Input:");
        check_buf(*engine_needs_plugin, "ao");
        check_buf(*engine_needs_plugin, "depth");
        check_buf(*engine_needs_plugin, "color");
        check_buf(*engine_needs_plugin, "motion");
        check_buf(*engine_needs_plugin, "temporal_ao");
        check_buf(*engine_needs_plugin, "temporal_embed");

        LOG_DEBUG("");
        LOG_DEBUG("Output:");
        check_buf(*engine_needs_plugin, "final_output");
        check_buf(*engine_needs_plugin, "denoised");
        check_buf(*engine_needs_plugin, "new_temporal_ao");
        check_buf(*engine_needs_plugin, "new_temporal_embed");

        LOG_DEBUG("");
        LOG_DEBUG("");
    }
};

#undef checkCudaErrors
#undef checkCusolverErrors

} // namespace Moer::Render::Raster