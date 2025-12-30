#pragma once
#include "misc/Singleton.h"
#include "rhi/RHIResource.h"
#include <filesystem>

namespace Moer {
class EPixelFormat;

namespace Render {
class RenderDevice;
class CommandList;
}; // namespace Render

/**
 * @brief TextureLibrary manages scene textures:
 *        - Load textures from disk (PNG, JPG, KTX, DDS, etc)
 *        - Upload texture data to GPU
 *        - Cache textures to avoid duplicate loading
 *        - Support mipmap generation and automatic format conversion
 */
class RENDER_API TextureLibrary : public Singleton<TextureLibrary> {
public:
    TextureLibrary();
    ~TextureLibrary();

    /**
     * @brief Load texture from file and immediately upload to GPU
     * @param _device GPU render device
     * @param _cmd_list Command list to submit upload commands
     * @param _path File path to texture asset
     * @param _name Optional name for debugging/caching
     * @return GPU texture reference, or nullptr if failed
     */
    Render::TextureRef LoadTexture(
        Render::RenderDevice&        _device,
        Render::CommandList&         _cmd_list,
        const std::filesystem::path& _path,
        std::string_view             _name = ""
    );

    /**
     * @brief Load texture from memory buffer and upload to GPU
     * @param _device GPU render device
     * @param _cmd_list Command list to submit upload commands
     * @param _data Raw image data buffer
     * @param _size Buffer size in bytes
     * @param _name Optional name for debugging/caching
     * @return GPU texture reference, or nullptr if failed
     */
    Render::TextureRef LoadTextureFromMemory(
        Render::RenderDevice& _device,
        Render::CommandList&  _cmd_list,
        const uint8*          _data,
        size_t                _size,
        std::string_view      _name = ""
    );

    /**
     * @brief Get cached texture by name
     * @return GPU texture reference, or nullptr if not found
     */
    Render::TextureRef GetTexture(std::string_view _name) const;

    /**
     * @brief Check if texture is already cached
     */
    bool HasTexture(std::string_view _name) const;

    /**
     * @brief Clear all cached textures
     */
    void Clear();

    /**
     * @brief Get statistics about cached textures
     */
    std::string GetStatistics() const;

private:
    struct TextureEntry {
        Render::TextureRef texture;
        uint32             bindless_handle;
        std::string        file_path;
        size_t             gpu_memory_bytes;
    };

    // Name -> Texture mapping for caching
    UnorderedMap<std::string, TextureEntry> m_textures;

    // Total GPU memory used
    size_t m_total_gpu_memory{0};

    /**
     * @brief Internal: Create GPU texture from loaded image data
     */
    Render::TextureRef CreateGPUTexture(
        Render::RenderDevice& _device,
        uint32                _width,
        uint32                _height,
        uint32                _mips,
        uint32                _layers,
        EPixelFormat          _format,
        std::string_view      _name
    );

    /**
     * @brief Internal: Upload image data to GPU texture via command list
     */
    bool UploadToGPU(
        Render::CommandList& _cmd_list,
        void*                _data,
        size_t               _data_size,
        uint32               _width,
        uint32               _height,
        uint32               _mips,
        EPixelFormat         _format,
        Render::TextureRef   _gpu_texture
    );

    /**
     * @brief Compute unique name from file path
     */
    static std::string ComputeTextureName(const std::filesystem::path& _path);
};

} // namespace Moer