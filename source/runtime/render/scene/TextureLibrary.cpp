#include "TextureLibrary.h"
#include "log/LogSystem.h"

namespace Moer {

TextureLibrary::TextureLibrary() : m_total_gpu_memory(0) {
    LOG_INFO("TextureLibrary initialized");
}

TextureLibrary::~TextureLibrary() {
    Clear();
}

Render::TextureRef TextureLibrary::LoadTexture(
    Render::RenderDevice&        _device,
    Render::CommandList&         _cmd_list,
    const std::filesystem::path& _path,
    std::string_view             _name
) {
    // Compute texture name from path if not provided
    std::string texture_name = _name.empty() ? ComputeTextureName(_path) : std::string(_name);

    // Check if already cached
    if (HasTexture(texture_name)) {
        LOG_INFO("Texture cache hit: {}", texture_name);
        return GetTexture(texture_name);
    }

    // Load image from file
    Resource::ImageReadDesc image_data = Resource::ImageIO::ReadFromFile(
        _path,
        4, // desired channels (RGBA)
        PF_R8G8B8A8_UNORM
    );

    if (!image_data.IsValid()) {
        LOG_ERROR("Failed to load texture from file: {}", _path.string());
        return nullptr;
    }

    // Create GPU texture
    Render::TextureRef gpu_texture = CreateGPUTexture(
        _device,
        image_data.width,
        image_data.height,
        image_data.mips,
        image_data.layers,
        image_data.format,
        texture_name
    );

    if (!gpu_texture) {
        LOG_ERROR("Failed to create GPU texture: {}", texture_name);
        image_data.data_callback(image_data.data);
        return nullptr;
    }

    // Upload texture data to GPU
    if (!UploadToGPU(
            _cmd_list,
            image_data.data,
            image_data.data_size,
            image_data.width,
            image_data.height,
            image_data.mips,
            image_data.format,
            gpu_texture
        )) {
        LOG_ERROR("Failed to upload texture to GPU: {}", texture_name);
        image_data.data_callback(image_data.data);
        return nullptr;
    }

    // Cache the texture
    TextureEntry entry{};
    entry.texture          = gpu_texture;
    entry.file_path        = _path.string();
    entry.gpu_memory_bytes = image_data.data_size;

    m_textures[texture_name] = entry;
    m_total_gpu_memory += entry.gpu_memory_bytes;

    LOG_INFO(
        "Texture loaded successfully: {} ({} bytes, {}x{}, {} mips)",
        texture_name,
        image_data.data_size,
        image_data.width,
        image_data.height,
        image_data.mips
    );

    // Cleanup CPU-side data
    image_data.data_callback(image_data.data);

    return gpu_texture;
}

Render::TextureRef TextureLibrary::LoadTextureFromMemory(
    Render::RenderDevice& _device,
    Render::CommandList&  _cmd_list,
    const uint8*          _data,
    size_t                _size,
    std::string_view      _name
) {
    std::string texture_name = _name.empty() ?
                                   "MemoryTexture_" + std::to_string(reinterpret_cast<uintptr_t>(_data)) :
                                   std::string(_name);

    // Check if already cached
    if (HasTexture(texture_name)) {
        LOG_INFO("Texture cache hit (memory): {}", texture_name);
        return GetTexture(texture_name);
    }

    // Load image from memory
    Resource::ImageReadDesc image_data = Resource::ImageIO::ReadFromMemory(
        _data,
        _size,
        4 // desired channels (RGBA)
    );

    if (!image_data.IsValid()) {
        LOG_ERROR("Failed to load texture from memory (size: {})", _size);
        return nullptr;
    }

    // Create GPU texture
    Render::TextureRef gpu_texture = CreateGPUTexture(
        _device,
        image_data.width,
        image_data.height,
        image_data.mips,
        image_data.layers,
        image_data.format,
        texture_name
    );

    if (!gpu_texture) {
        LOG_ERROR("Failed to create GPU texture from memory");
        image_data.data_callback(image_data.data);
        return nullptr;
    }

    // Upload texture data to GPU
    if (!UploadToGPU(
            _cmd_list,
            image_data.data,
            image_data.data_size,
            image_data.width,
            image_data.height,
            image_data.mips,
            image_data.format,
            gpu_texture
        )) {
        LOG_ERROR("Failed to upload texture to GPU from memory");
        image_data.data_callback(image_data.data);
        return nullptr;
    }

    // Cache the texture
    TextureEntry entry{};
    entry.texture          = gpu_texture;
    entry.file_path        = "memory://";
    entry.gpu_memory_bytes = image_data.data_size;

    m_textures[texture_name] = entry;
    m_total_gpu_memory += entry.gpu_memory_bytes;

    LOG_INFO(
        "Texture loaded from memory: {} ({} bytes, {}x{})",
        texture_name,
        image_data.data_size,
        image_data.width,
        image_data.height
    );

    // Cleanup CPU-side data
    image_data.data_callback(image_data.data);

    return gpu_texture;
}

Render::TextureRef TextureLibrary::GetTexture(std::string_view _name) const {
    auto it = m_textures.find(std::string(_name));
    if (it != m_textures.end()) {
        return it->second.texture;
    }
    return nullptr;
}

bool TextureLibrary::HasTexture(std::string_view _name) const {
    return m_textures.find(std::string(_name)) != m_textures.end();
}

void TextureLibrary::Clear() {
    m_textures.clear();
    m_total_gpu_memory = 0;
    LOG_INFO("TextureLibrary cleared");
}

std::string TextureLibrary::GetStatistics() const {
    std::string stats;
    stats += "TextureLibrary Statistics:\n";
    stats += "  Cached Textures: " + std::to_string(m_textures.size()) + "\n";
    stats += "  Total GPU Memory: " + std::to_string(m_total_gpu_memory / (1024 * 1024)) + " MB\n";
    return stats;
}

Render::TextureRef TextureLibrary::CreateGPUTexture(
    Render::RenderDevice& _device,
    uint32                _width,
    uint32                _height,
    uint32                _mips,
    uint32                _layers,
    EPixelFormat          _format,
    std::string_view      _name
) {
    // Determine texture dimension based on layer count
    ETextureDimension dimension = _layers > 1 ? ETextureDimension::TEX_2D_ARRAY : ETextureDimension::TEX_2D;

    // Create GPU texture with shader read and transfer dst flags
    ETextureUsageFlags usage_flags = ETextureUsageFlags::SHADER_READ | ETextureUsageFlags::TRANSFER_DST;

    Render::TextureRef gpu_texture = _device.CreateTexture(
        _name, Render::Extent3D{_width, _height, 1}, _format, usage_flags, _mips, _layers
    );

    if (!gpu_texture) {
        LOG_ERROR("Failed to create GPU texture: {}", _name);
        return nullptr;
    }

    LOG_INFO(
        "GPU texture created: {} ({}x{}, {} mips, {} layers, format={})",
        _name,
        _width,
        _height,
        _mips,
        _layers,
        (int)_format
    );

    return gpu_texture;
}

bool TextureLibrary::UploadToGPU(
    Render::CommandList& _cmd_list,
    void*                _data,
    size_t               _data_size,
    uint32               _width,
    uint32               _height,
    uint32               _mips,
    EPixelFormat         _format,
    Render::TextureRef   _gpu_texture
) {
    if (!_gpu_texture || !_data || _data_size == 0) {
        LOG_ERROR("Invalid parameters for GPU upload");
        return false;
    }

    // Create texture view for the GPU texture and submit upload command
    Render::TextureView tex_view(_gpu_texture);
    tex_view.extent    = {_width, _height, 1};
    tex_view.mip_level = 0;
    tex_view.num_mips  = _mips;

    // Submit upload command via command list
    // The span wraps the entire buffer
    auto data_span = std::span<Moer::byte>(static_cast<Moer::byte*>(_data), _data_size);
    _cmd_list.CopyFrom(data_span, tex_view, "TextureUpload");

    LOG_INFO("Submitted texture upload command ({} bytes)", _data_size);
    return true;
}

std::string TextureLibrary::ComputeTextureName(const std::filesystem::path& _path) {
    // Use filename without extension as the texture name
    return _path.stem().string();
}

} // namespace Moer
