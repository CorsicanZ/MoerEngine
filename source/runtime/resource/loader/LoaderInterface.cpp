#include "loader/LoaderInterface.h"
#include "loader/cache/SceneCache.h"

#include "ResourceAPI.h"
#include "loader/gltf/Parser.h"
#include "loader/jsonscene/JsonSceneParser.h"
#include "loader/ply/Ply.h"
#include "log/LogSystem.h"
#include "scene/Scene.h"
#include "scene/material/MaterialInstance.h"
#include "taskgraph/TaskGraph.h"

#include <filesystem>
namespace Moer::Resource {
using namespace Moer::ECS;

using LoadFunction = std::function<bool(const std::filesystem::path&, Scene&)>;
static Moer::Map<std::string, LoadFunction> scene_load_function_maps = {
    {"gltf", Gltf::Parser::LoadSceneFromFile},
    {"glb", Gltf::Parser::LoadSceneFromFile},
    {"fbx", Gltf::Parser::LoadSceneFromFile},
    {"obj", Gltf::Parser::LoadSceneFromFile},
    {"json", Gltf::Parser::LoadSceneFromFile}
};

void LoadFromFile(const std::filesystem::path& _file_path, Scene& _scene) {
    auto ext = _file_path.string().substr(_file_path.string().find_last_of(".") + 1);
    if (scene_load_function_maps.find(ext) == scene_load_function_maps.end()) {
        LOG_ERROR("Unsupported file format: {}", _file_path.extension().string());
        return;
    }
    AsyncSceneLoadInfoRef load_info = MoerNew(AsyncSceneLoadInfo)();
    load_info->b_valid              = true;
    load_info->progress.store(0);
    _scene.RegisterAsyncLoadInfo(load_info);
    LambdaTask::Dispatch([_file_path, &_scene, load_info, ext]() {
        if (scene_load_function_maps[ext](_file_path, _scene)) {
            LOG_INFO("Raw data is loaded to LoadedSceneDesc successfully, converting to Scene...");
            // SceneCache::ConvertToScene(*scene_data, _scene, true);
            load_info->progress.store(1);
            LOG_INFO("Scene loaded successfully from file: {}", _file_path.string());
        }
    });
}

void LoaderInterface::LoadSceneFromFileAsync(
    const std::filesystem::path& _file_path,
    Scene&                       _scene
) noexcept {
    auto file_path_str = _file_path.string();
    LOG_INFO("Loading scene from file: {}", file_path_str);
    if (_file_path.string().ends_with(".ply")) {
        assert(false && "Ply file is not supported yet");
        // auto gs_scene = PlyLoader::LoadSceneFromFile(_file_path);
        // scene->SetBuffer(EGpuSceneResource::GaussianSplattingVertex, gs_scene->GetBuffer(EGpuSceneResource::GaussianSplattingVertex));
    } else {
        if (false) {
            LambdaTask::Dispatch([_file_path, &_scene]() {
                try {
                    SceneCache::LoadSceneFromCache(_file_path, _scene);
                    LOG_INFO("Scene loaded successfully from cache: {}", _file_path.string());
                } catch (const std::exception& e) {
                    LOG_ERROR("Failed to load scene from cache: {} retrying to load from file", e.what());
                    _scene.ResetAsyncLoadInfo();
                    LoadFromFile(_file_path, _scene);
                }
            });
        } else {
            LoadFromFile(_file_path, _scene);
        }
    }
}

} // namespace Moer::Resource