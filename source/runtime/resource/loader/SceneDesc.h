#pragma once
#include "scene/Scene.h"
#include "scene/camera/Camera.h"
#include "scene/light/LightComponent.h"
#include "scene/material/material.h"
#include "serialize/Serializer.h"
#include "shaderheaders/shared/Geometry.h"


#include <filesystem>
namespace Moer {
struct MatInstanceTextureInfo {
    Moer::Array<std::pair<std::string, std::string>> textures;
};

struct TextureData {
    uint32_t             width{0}, height{0}, layers{1}, mips{1}, channal{4}, data_size{0};
    EPixelFormat         format{PF_UNDEFINED};
    Moer::Array<uint8_t> data;
    Array<uint32_t>      mip_offsets = {0};
    Array<Extent3D>      mip_extents;

    OutputStream& operator<<(OutputStream& _stream) const {
        _stream << width << height << layers << mips << channal << data_size << format << data << mip_offsets
                << mip_extents;
        return _stream;
    }

    InputStream& operator>>(InputStream& _stream) {
        _stream >> width >> height >> layers >> mips >> channal >> data_size >> format >> data >>
            mip_offsets >> mip_extents;
        return _stream;
    }
};
} // namespace Moer