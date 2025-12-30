#pragma once
#include "ResourceAPI.h"
#include <filesystem>
#include <future>
#include <memory>
#include <scene/Scene.h>
#include <string>

namespace Moer::Resource::JsonScene {
using Path = std::filesystem::path;
class JsonSceneParser {
public:
    JsonSceneParser() noexcept;
    ~JsonSceneParser() noexcept;

    static RESOURCE_API bool LoadSceneFromFile(const Path& _abs_scn_json_path, Scene& _scene) noexcept;

private:
    class Impl;
    Impl* m_impl = nullptr;
};
} // namespace Moer::Resource::JsonScene