#include "loader/cache/SceneCache.h"

#include "config/ConfigManager.h"
#include "misc/Timer.h"

#include "scene/Scene.h"
#include "taskgraph/TaskGraph.h"

#include <filesystem>
#include <fstream>

#include <serialize/Serializer.h>

namespace Moer::ECS {

static std::filesystem::path RemapScenePath(const std::filesystem::path& _path) {
    long long time = std::filesystem::exists(_path) ?
                         std::filesystem::last_write_time(_path).time_since_epoch().count() :
                         0;
    auto      cache_path =
        (ConfigManager::GetInstance().GetCachePath() /
         std::format("{}_{}", _path.filename().generic_string(), uint64(time)));
    return cache_path.generic_string() + ".MOERSCENE";
}

void SceneCache::FromFile(const std::filesystem::path& _path, Scene& _scene) {
    using namespace Moer;
    Timer timer;
    timer.Start();
    std::ifstream fs(_path, std::ios::binary);

    InputStream stream(fs);

    // LoadedSceneDesc scene_data;

    // ReadSceneGeomInfo(stream, scene_data);
    // ReadSceneTextures(stream, scene_data);
    // ReadSceneMaterial(stream, scene_data);
    // ReadSceneUtils(stream, scene_data);

    // ConvertToScene(scene_data, _scene, false);
    timer.Stop();
    LOG_INFO("Load Scene Cache Time(ms): {}", timer.ElapsedMilliseconds());
}

bool SceneCache::HasValidCache(const std::filesystem::path& _path) {
    auto cache_path = RemapScenePath(_path);
    return std::filesystem::exists(cache_path);
}

// void SceneCache::ReadSceneTextures(FInputStream& stream, LoadedSceneDesc& sceneData) {
//     size_t texture_count;
//     stream >> texture_count;

//     for (uint i = 0; i < texture_count; ++i) {
//         std::string name;
//         stream >> name;

//         TextureData texture_data;

//         stream >> texture_data;
//         sceneData.m_textures[name] = std::move(texture_data);
//     }
// }

// void SceneCache::ReadSceneUtils(InputStream& stream, LoadedSceneDesc& _scene_desc) {
//     // read cameras
//     size_t camera_count;
//     stream >> camera_count;
//     _scene_desc.m_camera_array.reserve(camera_count);
//     for (int i = 0; i < camera_count; ++i) {
//         CameraRef camera = MoerNew(Camera);
//         // stream.Read(camera, sizeof(Camera));
//         stream >> *camera;
//         _scene_desc.m_camera_array.push_back(camera);
//     }

//     // read lights
//     size_t light_count;
//     stream >> light_count;
//     _scene_desc.m_light_array.reserve(light_count);
//     for (int i = 0; i < light_count; ++i) {
//         LightComponentRef light = LightComponent::ReadFromStream(stream);
//         _scene_desc.m_light_array.push_back(light);
//     }
// }

// void SceneCache::ReadSceneGeomInfo(FInputStream& _stream, LoadedSceneDesc& _scene_desc) {
//     _stream >>
//         _scene_desc.m_mesh_instance_array >>
//         _scene_desc.m_mesh_object_array >>
//         _scene_desc.m_geometry_object_array >>
//         _scene_desc.m_transform_array >>
//         _scene_desc.m_mesh_data_array;
// }

// void SceneCache::ReadSceneMaterial(InputStream& _stream, LoadedSceneDesc& _scene_desc) {
//     // size_t material_count;
//     // _stream >> material_count;

//     // for (int i = 0; i < material_count; ++i) {
//     //     std::string material_name;
//     //     _stream >> material_name;

//     //     // block
//     //     BufferInterfaceBlock block;
//     //     {
//     //         _stream >> block.m_name;

//     //         _stream >> block.m_size;
//     //         _stream >> block.m_alignment;
//     //         _stream >> block.m_target;
//     //         _stream >> block.m_qualifiers;

//     //         _stream >> block.m_field_info_list;

//     //         size_t info_count;
//     //         _stream >> info_count;
//     //         for (int j = 0; j < info_count; ++j) {
//     //             std::string name;
//     //             _stream >> name;
//     //             uint32_t offset;
//     //             _stream >> offset;
//     //             block.m_info_map[block.m_field_info_list[offset].name] = offset;
//     //         }
//     //     }
//     //     // sampler
//     //     TextureInterfaceBlock sampler;
//     //     {
//     //         _stream >> sampler.m_name;

//     //         _stream >> sampler.m_sampler_info_list;

//     //         size_t info_count;
//     //         _stream >> info_count;
//     //         for (int j = 0; j < info_count; ++j) {
//     //             std::string name;
//     //             _stream >> name;
//     //             uint8_t offset;
//     //             _stream >> offset;
//     //             sampler.m_info_map[sampler.m_sampler_info_list[offset].name] = offset;
//     //         }
//     //     }

//     //     MaterialRef material = MoerNew(Material);
//     //     material->SetBufferInterfaceBlock(block);
//     //     material->SetSamplerInterfaceBlock(sampler);
//     //     material->SetName(material_name);

//     //     _scene_desc.m_materials[material_name] = material;
//     // }

//     // size_t material_instance_count;
//     // _stream >> material_instance_count;
//     // for (int i = 0; i < material_instance_count; ++i) {
//     //     std::string material_instance_name;
//     //     _stream >> material_instance_name;
//     //     uint inst_idx;
//     //     _stream >> inst_idx;
//     //     _scene_desc.m_material_instance_indexes[material_instance_name] = inst_idx;
//     // }
//     // _scene_desc.m_material_instances.resize(material_instance_count);

//     // for (int i = 0; i < material_instance_count; ++i) {
//     //     std::string material_instance_name;
//     //     _stream >> material_instance_name;
//     //     std::string mat_name;
//     //     _stream >> mat_name;
//     //     auto   material_instance = _scene_desc.m_materials[mat_name]->CreateInstance();
//     //     size_t texture_param_count;
//     //     _stream >> texture_param_count;
//     //     for (int j = 0; j < texture_param_count; ++j) {
//     //         std::string param_name;
//     //         _stream >> param_name;
//     //         std::string texture_name;
//     //         _stream >> texture_name;
//     //         _scene_desc.m_mat_instance_textures[material_instance_name].textures.emplace_back(param_name, texture_name);
//     //     }

//     //     auto unfirom_buffer_size = material_instance->GetUniformBuffer().GetSize();
//     //     auto buffer_data         = Moer::Array<uint8_t>(unfirom_buffer_size);
//     //     _stream >> buffer_data;
//     //     material_instance->SetUnifomBuffer(buffer_data.data(), unfirom_buffer_size);
//     //     material_instance->SetName(material_instance_name);
//     //     _scene_desc.m_material_instances[_scene_desc.m_material_instance_indexes[material_instance_name]] = material_instance;
//     // }
// }

// void SceneCache::WriteSceneGeomInfo(FOutputStream& _stream, const LoadedSceneDesc& _scene_desc) {
//     _stream << _scene_desc.m_mesh_instance_array << _scene_desc.m_mesh_object_array << _scene_desc.m_geometry_object_array << _scene_desc.m_transform_array << _scene_desc.m_mesh_data_array;
// }

// void SceneCache::WriteSceneMaterial(FOutputStream& _stream, const LoadedSceneDesc& _scene_desc) {
//     // _stream << _scene_desc.m_materials.size();
//     // for (auto& material : _scene_desc.m_materials) {
//     //     _stream << material.first;
//     //     auto& sampler_info = material.second->GetSamplerInterfaceBlock();
//     //     auto& buffer_info  = material.second->GetBufferInterfaceBlock();

//     //     {
//     //         _stream << buffer_info.m_name;
//     //         _stream << buffer_info.m_size;
//     //         _stream << buffer_info.m_alignment;
//     //         _stream << buffer_info.m_target;
//     //         _stream << buffer_info.m_qualifiers;

//     //         _stream << buffer_info.m_field_info_list;

//     //         _stream << buffer_info.m_info_map.size();
//     //         for (auto& info : buffer_info.m_info_map) {
//     //             _stream << info.first;
//     //             _stream << info.second;
//     //         }
//     //     }

//     //     {
//     //         _stream << sampler_info.m_name;
//     //         _stream << sampler_info.m_sampler_info_list;
//     //         _stream << sampler_info.m_info_map.size();
//     //         for (auto& info : sampler_info.m_info_map) {
//     //             _stream << info.first;
//     //             _stream << info.second;
//     //         }
//     //     }
//     // }

//     // _stream << _scene_desc.m_material_instances.size();
//     // for (auto [name, index] : _scene_desc.m_material_instance_indexes) {
//     //     _stream << name << index;
//     // }
//     // for (auto [name, index] : _scene_desc.m_material_instance_indexes) {
//     //     auto& material_instance = _scene_desc.m_material_instances.at(index);
//     //     _stream << name;
//     //     _stream << material_instance->GetMaterial()->GetName();

//     //     if (_scene_desc.m_mat_instance_textures.contains(name)) {
//     //         const MatInstanceTextureInfo& mat_instance_texture_info = _scene_desc.m_mat_instance_textures.at(name);
//     //         _stream << mat_instance_texture_info.textures.size();

//     //         for (auto& texture : mat_instance_texture_info.textures) {
//     //             _stream << texture.first;
//     //             _stream << texture.second;
//     //         }

//     //     } else {
//     //         // Because in ReadSceneMaterial(), we expect the texture_param_count to be a size_t
//     //         // Here we need to manual cast 0 to size_t to ensure it is written in 8 bytes instead of 4
//     //         // stream.write(static_cast<size_t>(0));
//     //         _stream << 0ull;
//     //     }
//     //     _stream << std::span<byte>((byte*)material_instance->GetUniformBuffer().GetData(), material_instance->GetUniformBuffer().GetSize());
//     // }
// }

// void SceneCache::WriteSceneTextures(FOutputStream& _stream, const LoadedSceneDesc& _scene_desc) {
//     _stream << _scene_desc.m_textures.size();

//     for (auto& texture : _scene_desc.m_textures) {
//         _stream << texture.first;
//         _stream << texture.second;
//     }
// }

// void SceneCache::WriteSceneUtils(OutputStream& _stream, const LoadedSceneDesc& _scene_desc) {
//     // write cameras
//     _stream << _scene_desc.m_camera_array.size();
//     for (auto& camera : _scene_desc.m_camera_array) {
//         _stream << *camera;
//     }

//     // write lights
//     _stream << _scene_desc.m_light_array.size();
//     for (auto& light : _scene_desc.m_light_array) {
//         light->WriteToStream(_stream);
//     }
// }

// size_t HashSceneData(const LoadedSceneDesc& _scene_desc) {
//     size_t hash = 0;
//     return hash;
// }

// void SceneCache::Cache(const LoadedSceneDesc& _scene_desc, size_t _key) {
//     std::filesystem::path path = RemapScenePath(_scene_desc.m_path);
//     if (!std::filesystem::exists(path.parent_path())) {
//         std::filesystem::create_directories(path.parent_path());
//     }
//     std::ofstream fs(path, std::ios::binary);
//     OutputStream  stream(fs);
//     WriteSceneGeomInfo(stream, _scene_desc);
//     WriteSceneTextures(stream, _scene_desc);
//     WriteSceneMaterial(stream, _scene_desc);
//     WriteSceneUtils(stream, _scene_desc);
// }

// /**
//  * @brief convert from loaded cpu scene data to ECS desc and buffer&texture
//  *
//  * @param _scene_desc
//  * @param _scene
//  * @param _need_cache
//  */
// void SceneCache::ConvertToScene(LoadedSceneDesc& _scene_desc, Scene* _scene, bool _need_cache) {
//     using namespace Moer::Render;
//     using namespace Moer;

//     const auto& mesh_instance_array = _scene_desc.m_mesh_instance_array;
//     const auto& mesh_object_array   = _scene_desc.m_mesh_object_array;
//     const auto& geom_object_array   = _scene_desc.m_geometry_object_array;
//     const auto& transform_array     = _scene_desc.m_transform_array;
//     const auto& mesh_data_array     = _scene_desc.m_mesh_data_array;
//     /******************* create instance entities */
//     for (auto& mesh_instance : mesh_instance_array) {
//         auto entity = EntityManager::Get().Create();
//         _scene->AddEntity(entity);
//         RenderableManager::Get().CreateMeshInstance(entity);
//         RenderableManager::Get().SetMeshObject(entity, mesh_object_array[mesh_instance.mesh_id]);
//         RenderableManager::Get().SetInstanceID(entity, mesh_instance.instance_id);
//         Array<MaterialInstanceRef> material_instances;

//         auto mesh_object = mesh_object_array[mesh_instance.mesh_id];

//         material_instances.reserve(mesh_object->geom_cnt);
//         for (const auto& geo : std::span(geom_object_array).subspan(mesh_object->geom_id, mesh_object->geom_cnt)) {
//             // material_instances.emplace_back(_scene_desc.m_material_instances[geo->material_id]);
//         }
//         RenderableManager::Get().SetMaterialInstances(entity, std::move(material_instances));

//         TransformManager::Get().Create(entity);
//         TransformManager::Get().Set(entity, transform_array[mesh_instance.transform_id]);
//     }

//     assert(mesh_instance_array.size() != 0 && "Mesh instances should not be empty");

//     /******************* create camera entities */
//     for (auto& camera : _scene_desc.m_camera_array) {
//         auto entity = EntityManager::Get().Create();
//         CameraManager::Get().Put(entity, camera);
//         _scene->AddCamera(entity);
//     }

//     /******************* create light entities */
//     for (auto& light : _scene_desc.m_light_array) {
//         auto entity = EntityManager::Get().Create();
//         LightComponentManager::Get().Put(entity, light);
//         _scene->AddLight(entity);
//     }

//     /******************* copyqueue, copy mesh buffers, texture images, material buffers, light buffers to GPU */
//     _scene->FillSceneDesc(geom_object_array, mesh_data_array);

//     /******************* export textures and buffers from the gpu copy queue to the gpu graphics queue */
//     auto&                        device = Render::RenderDevice::Get();
//     CommandList                  cmd_list{};
//     auto&                        copy_queue = device.GetCopyQueue();
//     Array<Render::ExportTexture> export_textures;
//     // export_textures.reserve(textures.size());

//     // for (auto& texture : textures) {
//     //     export_textures.push_back({texture.second->GetView(), ETextureState::SAMPLE});
//     // }

//     Array<Render::ExportBuffer> export_buffers;
//     export_buffers.reserve(_scene->GetIOPendingBuffers().size());

//     for (auto& buffer : _scene->GetIOPendingBuffers()) {
//         export_buffers.push_back({buffer->GetView(), EBufferState::UNORDERED_ACCESS});
//     }

//     cmd_list.ExportResourcesToQueue(EQueueType::Graphics, std::move(export_textures), std::move(export_buffers));

//     auto copy_handle = copy_queue.GetFenceHandle();

//     //wait all texture export done
//     auto evt = copy_queue.Execute(cmd_list.Submit().Wait(copy_handle, copy_handle->GetValue()));
//     copy_queue.Sync(evt.timeline);

//     // Cache the scene data
//     // Tip: I move this function to the end of the function, to avoid storing wrong cache to disk.
//     //      Wrong cache may be caused by out-date loader code.
//     //      When loading wrong cache, the engine will crash with no information.
//     auto hash    = HashSceneData(_scene_desc);
//     bool updated = true;
//     if (updated && _need_cache) {
//         Cache(_scene_desc, hash);
//     }
// }

void SceneCache::LoadSceneFromCacheAsync(const std::filesystem::path& _path, Scene& _scene) {
    LambdaTask::Dispatch([_path, &_scene]() {
        AsyncSceneLoadInfoRef load_info = MoerNew(AsyncSceneLoadInfo)();
        load_info->b_valid              = true;
        load_info->progress.store(0);
        _scene.RegisterAsyncLoadInfo(load_info);
        FromFile(RemapScenePath(_path), _scene);
        // load_info->scene = scene_data;
        // Scene::SetCurrentScene(load_info->scene);
        load_info->progress.store(1);
    });
}

void SceneCache::LoadSceneFromCache(const std::filesystem::path& _path, Scene& _scene) {
    AsyncSceneLoadInfoRef load_info = MoerNew(AsyncSceneLoadInfo)();
    load_info->b_valid              = true;
    load_info->progress.store(0);
    _scene.RegisterAsyncLoadInfo(load_info);
    FromFile(RemapScenePath(_path), _scene);
    // load_info->scene = scene_data;
    // Scene::SetCurrentScene(load_info->scene);
    load_info->progress.store(1);
}

} // namespace Moer::ECS