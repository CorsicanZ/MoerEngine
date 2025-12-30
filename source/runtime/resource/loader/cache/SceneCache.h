#pragma once
#include "ResourceAPI.h"
#include "scene/Scene.h"
#include "serialize/Serializer.h"
#include <filesystem>

namespace Moer::Resource {
class SceneCache {
public:
    using FInputStream  = Moer::InputStream;
    using FOutputStream = Moer::OutputStream;

    static RESOURCE_API void FromFile(const std::filesystem::path& _path, Scene& _scene);
    static RESOURCE_API bool HasValidCache(const std::filesystem::path& _path);
    // static RESOURCE_API void ConvertToScene(LoadedSceneDesc& _scene_desc, ECS::Scene* _scene, bool _need_cache = true);
    static RESOURCE_API void LoadSceneFromCacheAsync(const std::filesystem::path& _path, Scene& _scene);
    static RESOURCE_API void LoadSceneFromCache(const std::filesystem::path& _path, Scene& _scene);
    //
protected:
    // static RESOURCE_API void ReadSceneGeomInfo(FInputStream& _stream, LoadedSceneDesc& _scene_desc);
    // static RESOURCE_API void ReadSceneMaterial(FInputStream& _stream, LoadedSceneDesc& _scene_desc);
    // static RESOURCE_API void ReadSceneTextures(FInputStream& _stream, LoadedSceneDesc& _scene_desc);
    // static RESOURCE_API void ReadSceneUtils(FInputStream& _stream, LoadedSceneDesc& _scene_desc);

    // static RESOURCE_API void WriteSceneMaterial(FOutputStream& _stream, const LoadedSceneDesc& _scene_desc);
    // static RESOURCE_API void WriteSceneGeomInfo(FOutputStream& _stream, const LoadedSceneDesc& _scene_desc);
    // static RESOURCE_API void WriteSceneTextures(FOutputStream& _stream, const LoadedSceneDesc& _scene_desc);
    // static RESOURCE_API void WriteSceneUtils(FOutputStream& _stream, const LoadedSceneDesc& _scene_desc);

    // static RESOURCE_API void Cache(const LoadedSceneDesc& _scene_desc, size_t _key);
};
} // namespace Moer::Resource