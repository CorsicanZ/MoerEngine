#pragma once

#include "ResourceAPI.h"
#include <filesystem>

#include "scene/Scene.h"

namespace Moer::Resource::Gltf {

class Parser {
public:
    Parser() noexcept;
    ~Parser() noexcept;

    static RESOURCE_API bool
    LoadSceneFromFile(const std::filesystem::path& _file_path, Scene& _scene) noexcept;

private:
    struct Impl;
    Impl* m_impl = nullptr;
};
} // namespace Moer::Resource::Gltf