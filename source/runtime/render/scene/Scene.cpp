#include "scene/Scene.h"

#include "log/LogSystem.h"
#include "misc/STL.h"
#include "rhi/RHI.h"
#include "rhi/RHICommand.h"
#include "rhi/RHIResource.h"
#include "scene/Entity.h"
#include "scene/transform/TransformComponent.h"
#include <atomic>

namespace Moer {
using ECS::ComponentLibrary;
using ECS::ComponentManager;
using ECS::Entity;

// struct CpuSceneDesc {
//     Array<Render::GeometryData>                                               geometry_datas;
//     Array<Render::InstanceData>                                               instance_datas;
//     Array<Render::GeometryInstance>                                           geom_instances;
//     Array<UnorderedMap<VertexAttributesBitmask, Array<Render::VertexBuffer>>> vtx_views;
//     Array<UnorderedMap<VertexAttributesBitmask, Render::IndexBuffer>>         idx_views;
// };

// struct GpuSceneDesc {
//     StaticArray<Render::BufferRef, (uint32)EGpuSceneResource::Num> buffers;
//     UnorderedMap<std::string, SceneTexture>                        textures;
//     Render::BindlessArrayRef                                       bindless_array{nullptr};
// };

//////////////////////////////////////////////////////////////////////////
// Scene
//////////////////////////////////////////////////////////////////////////
bool Scene::IsDescendant(Entity _entity, Entity _ancestor) const {
    const HierarchyComponent* hier = hierarchy.GetComponent(_entity);
    while (hier != nullptr) {
        if (hier->parent_id == _ancestor)
            return true;
        hier = hierarchy.GetComponent(hier->parent_id);
    }
    return false;
}

Entity Scene::AddTransform(std::string_view _name) {
    auto entity = ECS::CreateEntity();

    names.Create(entity) = _name.data();

    transforms.Create(entity);

    return entity;
}

Entity Scene::AddCamera(std::string_view _name) {
    auto entity = ECS::CreateEntity();

    names.Create(entity) = _name.data();
    layers.Create(entity);
    transforms.Create(entity);

    cameras.Create(entity);

    return entity;
}

Entity Scene::AddLight(std::string_view _name) {
    auto entity = ECS::CreateEntity();

    names.Create(entity) = _name.data();
    layers.Create(entity);
    transforms.Create(entity);

    lights.Create(entity);

    return entity;
}

Entity Scene::AddMesh(std::string_view _name) {
    auto entity = ECS::CreateEntity();

    names.Create(entity) = _name.data();
    meshes.Create(entity);

    return entity;
}

Entity Scene::AddMeshInstance(std::string_view _name) {
    auto entity = ECS::CreateEntity();

    names.Create(entity) = _name.data();
    layers.Create(entity);
    transforms.Create(entity);
    mesh_instances.Create(entity);

    return entity;
}

Entity Scene::AddMaterial(std::string_view _name) {
    auto entity = ECS::CreateEntity();

    names.Create(entity) = _name.data();
    materials.Create(entity);

    return entity;
}

void Scene::Remove(ECS::Entity _entity, bool _recursive, bool _keep_sorted) {
    if (_recursive) {
        Array<Entity> entities_to_remove;
        for (size_t i = 0; i < hierarchy.GetCount(); ++i) {
            const HierarchyComponent& hier = hierarchy[i];
            if (hier.parent_id == _entity) {
                Entity child = hierarchy.GetEntity(i);
                entities_to_remove.push_back(child);
            }
        }
        for (auto& child : entities_to_remove) {
            Remove(child);
        }
    }

    for (auto& entry : m_component_library.entries) {
        if (_keep_sorted) {
            entry.second.component_manager->RemoveKeepSorted(_entity);
        } else {
            entry.second.component_manager->Remove(_entity);
        }

        if (m_topdown_hierarchy.count(_entity)) {
            m_topdown_hierarchy.erase(_entity);
        }
    }
}

void Scene::Attach(Entity _entity, Entity _parent, bool _child_already_in_local_space) {
    assert(_entity != _parent);

    if (hierarchy.Contains(_entity)) {
        Detach(_entity);
    }

    HierarchyComponent& parent_component = hierarchy.Create(_entity);
    parent_component.parent_id           = _parent;

    TransformComponent* transform_parent = transforms.GetComponent(_parent);
    TransformComponent* transform_child  = transforms.GetComponent(_entity);
    if (transform_parent != nullptr && transform_child != nullptr) {
        if (!_child_already_in_local_space) {
            auto b = transform_parent->transform.Inverse();
            transform_child->transform.AppendTransformation(b);
        }
        transform_child->transform.AppendTransformation(transform_parent->transform);
    }
}

void Scene::Detach(Entity _entity) {
    const HierarchyComponent* parent = hierarchy.GetComponent(_entity);

    if (parent != nullptr) {
        LayerComponent* layer = layers.GetComponent(_entity);
        if (layer != nullptr) {
            layer->propagation_mask = ~0;
        }

        hierarchy.Remove(_entity);
    }
}

void Scene::DetachChildren(Entity _parent) {
    for (auto i = 0; i < hierarchy.GetCount();) {
        if (hierarchy[i].parent_id == _parent) {
            Entity entity = hierarchy.GetEntity(i);
            Detach(entity);
        } else {
            ++i;
        }
    }
}

void Scene::TransformWorldToHierarchy(Entity _entity) {
    const HierarchyComponent* hier = hierarchy.GetComponent(_entity);
    if (hier == nullptr)
        return;
    const TransformComponent* transform_parent = transforms.GetComponent(hier->parent_id);
    if (transform_parent == nullptr)
        return;
    TransformComponent* transform_child = transforms.GetComponent(_entity);
    if (transform_child == nullptr)
        return;
    auto b = transform_parent->transform.Inverse();
    transform_child->transform.AppendTransformation(b);
}

void Scene::GatherChildren(Entity _parent, Array<Entity>& _children) const {
    for (auto i = 0; i < hierarchy.GetCount(); ++i) {
        Entity child = hierarchy.GetEntity(i);
        if (IsDescendant(child, _parent)) {
            Entity child = hierarchy.GetEntity(i);
            _children.push_back(child);
        }
    }
}

void Scene::Info() {
    LOG_INFO(
        "Scene Info: {} transforms, {} cameras, {} lights, {} meshes, {} mesh instances",
        transforms.GetCount(),
        cameras.GetCount(),
        lights.GetCount(),
        meshes.GetCount(),
        mesh_instances.GetCount()
    );
    for (auto& entry : m_component_library.entries) {
        LOG_INFO("Component: {}: {}", entry.first, entry.second.component_manager->GetCount());
    }
    LOG_INFO("BindlessArray: {}", bindless_array != nullptr ? "valid" : "invalid");
}

/* ECS-System tick*/
void Scene::TickTransform() {
    LambdaTask::Create([this]() {
        ParallelFor(transforms.GetCount(), [&](uint _index) {
            auto& transform = transforms[_index];
            // transform.Tick();
        });
    }).Dispatch();
}

void Scene::TickHierarchy() {
    LambdaTask::Create([this]() {
        ParallelFor(hierarchy.GetCount(), [&](uint _index) {
            auto& hier            = hierarchy[_index];
            auto  entity          = hierarchy.GetEntity(_index);
            auto* transform_child = transforms.GetComponent(entity);
            // todo
        });
    }).Dispatch();
}

void Scene::TickMesh() {}

void Scene::TickMaterial() {}

void Scene::TickMeshInstance() {}

void Scene::TickCamera() {}

void Scene::TickLight() {}

void Scene::Tick() {}

Render::BindlessArrayRef Scene::GetBindlessArray() const noexcept {
    return bindless_array;
}

// void Scene::FillSceneDesc(std::span<const GeometryObject> _geometry_object_array, std::span<const MeshData> _mesh_data_array) {
//     auto&               device = Render::RenderDevice::Get();
//     Render::CommandList cmd_list{};

//     auto  bindless_array = m_gpu_scene.bindless_array;
//     auto& copy_queue     = device.GetCopyQueue();

//     struct MeshDataHandle {
//         struct GpuMeshData {
//             Render::BufferRef vertex_buffer;
//             Render::BufferRef index_buffer;

//             int vtx_bdls_handle;
//             int idx_bdls_handle;
//         };
//         const MeshData* cpu_mesh_data;
//         GpuMeshData     gpu_mesh_data;
//     };

//     Array<MeshDataHandle> mesh_data_handle_array;
//     mesh_data_handle_array.reserve(_mesh_data_array.size());

//     // [GPU Resources]: create gpu vtx&idx buffers, copy from cpu to gpu
//     for (auto& mesh_data : _mesh_data_array) {
//         auto& handle         = mesh_data_handle_array.emplace_back();
//         handle.cpu_mesh_data = &mesh_data;
//         // vertex buffer
//         uint vertex_size = 0;

//         // clang-format off
//         auto position_buffer_size =
//             mesh_data.vertex_factory_buffers.HasAttribute(EVertexAttributes::VA_POSITION)
//             ? mesh_data.vertex_factory_buffers.GetBufferByteSize(EVertexAttributes::VA_POSITION)
//             : 0;

//         auto normal_buffer_size =
//             mesh_data.vertex_factory_buffers.HasAttribute(EVertexAttributes::VA_NORMAL)
//             ? mesh_data.vertex_factory_buffers.GetBufferByteSize(EVertexAttributes::VA_NORMAL)
//             : 0;

//         auto tangent_buffer_size =
//             mesh_data.vertex_factory_buffers.HasAttribute(EVertexAttributes::VA_TANGENT)
//             ? mesh_data.vertex_factory_buffers.GetBufferByteSize(EVertexAttributes::VA_TANGENT)
//             : 0;

//         auto texcoord0_buffer_size =
//             mesh_data.vertex_factory_buffers.HasAttribute(EVertexAttributes::VA_TEXCOORD0)
//             ? mesh_data.vertex_factory_buffers.GetBufferByteSize(EVertexAttributes::VA_TEXCOORD0)
//             : 0;

//         auto texcoord1_buffer_size =
//             mesh_data.vertex_factory_buffers.HasAttribute(EVertexAttributes::VA_TEXCOORD1)
//             ? mesh_data.vertex_factory_buffers.GetBufferByteSize(EVertexAttributes::VA_TEXCOORD1)
//             : 0;
//         // clang-format on

//         vertex_size += position_buffer_size;
//         vertex_size += normal_buffer_size;
//         vertex_size += tangent_buffer_size;
//         vertex_size += texcoord0_buffer_size;
//         vertex_size += texcoord1_buffer_size;

//         handle.gpu_mesh_data.vertex_buffer = device.CreateBuffer<byte>("Scene::soa_vertex_buffer", vertex_size, EBufferUsageFlags::VERTEX_BUFFER | EBufferUsageFlags::ACCELERATION_STRUCTURE | EBufferUsageFlags::UNORDERED_ACCESS);
//         if (position_buffer_size > 0) {
//             auto* position_buffer_ptr = mesh_data.vertex_factory_buffers.GetBufferData(EVertexAttributes::VA_POSITION);
//             cmd_list.CopyFrom(
//                 std::span<byte>((byte*)position_buffer_ptr, position_buffer_size),
//                 handle.gpu_mesh_data.vertex_buffer->GetView(0, mesh_data.GetAttributeRange(EVertexAttributes::VA_POSITION).size),
//                 "CopyFrom MeshBuffers position_buffer");
//         }
//         if (normal_buffer_size > 0) {
//             auto* normal_buffer_ptr = mesh_data.vertex_factory_buffers.GetBufferData(EVertexAttributes::VA_NORMAL);
//             cmd_list.CopyFrom(
//                 std::span<byte>((byte*)normal_buffer_ptr, normal_buffer_size),
//                 handle.gpu_mesh_data.vertex_buffer->GetView(mesh_data.GetAttributeRange(EVertexAttributes::VA_NORMAL).offset, mesh_data.GetAttributeRange(EVertexAttributes::VA_NORMAL).size),
//                 "CopyFrom MeshBuffers normal_buffer");
//         }
//         if (tangent_buffer_size > 0) {
//             auto* tangent_buffer_ptr = mesh_data.vertex_factory_buffers.GetBufferData(EVertexAttributes::VA_TANGENT);
//             cmd_list.CopyFrom(
//                 std::span<byte>((byte*)tangent_buffer_ptr, tangent_buffer_size),
//                 handle.gpu_mesh_data.vertex_buffer->GetView(mesh_data.GetAttributeRange(EVertexAttributes::VA_TANGENT).offset, mesh_data.GetAttributeRange(EVertexAttributes::VA_TANGENT).size),
//                 "CopyFrom MeshBuffers tangent_buffer");
//         }
//         if (texcoord0_buffer_size > 0) {
//             auto* texcoord0_buffer_ptr = mesh_data.vertex_factory_buffers.GetBufferData(EVertexAttributes::VA_TEXCOORD0);
//             cmd_list.CopyFrom(
//                 std::span<byte>((byte*)texcoord0_buffer_ptr, texcoord0_buffer_size),
//                 handle.gpu_mesh_data.vertex_buffer->GetView(mesh_data.GetAttributeRange(EVertexAttributes::VA_TEXCOORD0).offset, mesh_data.GetAttributeRange(EVertexAttributes::VA_TEXCOORD0).size),
//                 "CopyFrom MeshBuffers texcoord0_buffer");
//         }
//         if (texcoord1_buffer_size > 0) {
//             auto* texcoord1_buffer_ptr = mesh_data.vertex_factory_buffers.GetBufferData(EVertexAttributes::VA_TEXCOORD1);
//             cmd_list.CopyFrom(
//                 std::span<byte>((byte*)texcoord1_buffer_ptr, texcoord1_buffer_size),
//                 handle.gpu_mesh_data.vertex_buffer->GetView(mesh_data.GetAttributeRange(EVertexAttributes::VA_TEXCOORD1).offset, mesh_data.GetAttributeRange(EVertexAttributes::VA_TEXCOORD1).size),
//                 "CopyFrom MeshBuffers texcoord1_buffer");
//         }

//         // index buffer
//         handle.gpu_mesh_data.index_buffer = device.CreateBuffer<uint32>("Scene::index_buffer", mesh_data.indices.size(), EBufferUsageFlags::INDEX_BUFFER | EBufferUsageFlags::ACCELERATION_STRUCTURE | EBufferUsageFlags::UNORDERED_ACCESS);

//         cmd_list.CopyFrom(
//             std::span<byte>((byte*)mesh_data.indices.data(), mesh_data.indices.size() * sizeof(uint32_t)),
//             handle.gpu_mesh_data.index_buffer->GetView(),
//             "CopyFrom MeshBuffers index_buffer");

//         // bindless handle
//         handle.gpu_mesh_data.idx_bdls_handle = bindless_array->AllocateBuffer(handle.gpu_mesh_data.index_buffer->GetView());
//         handle.gpu_mesh_data.vtx_bdls_handle = bindless_array->AllocateBuffer(handle.gpu_mesh_data.vertex_buffer->GetView());

//         /******************* import mesh buffers from cpu-io to the copy queue */
//         EmplaceIOImportedBuffer(handle.gpu_mesh_data.vertex_buffer);
//         EmplaceIOImportedBuffer(handle.gpu_mesh_data.index_buffer);
//     }

//     // [CPU Resources]: fill cpu scene data: instance data, geometry data, geometry instance data
//     uint geom_instance_cnt = 0;
//     uint instance_count    = m_entities.entities.size();
//     for (auto& entity : m_entities.GetEntities()) {
//         const auto& mesh_object = RenderableManager::Get().GetMeshObject(entity);
//         geom_instance_cnt += mesh_object->geom_cnt;
//     }

//     LOG_INFO("UpdateGpuData, geometry_count:{}, instance_count:{}", geom_instance_cnt, instance_count);

//     m_cpu_scene.geometry_datas.resize(geom_instance_cnt);//reserve more space than needed
//     m_cpu_scene.instance_datas.resize(instance_count);
//     m_cpu_scene.geom_instances.resize(geom_instance_cnt);
//     m_cpu_scene.vtx_views.resize(instance_count);
//     m_cpu_scene.idx_views.resize(instance_count);

//     geom_instance_cnt = 0;
//     for (auto& entity : m_entities.GetEntities()) {
//         const auto& mesh_obj      = RenderableManager::Get().GetMeshObject(entity);
//         const auto  mat_instances = RenderableManager::Get().GetMaterialInstances(entity);
//         const auto  geom_array    = _geometry_object_array.subspan(mesh_obj->geom_id, mesh_obj->geom_cnt);
//         const auto  instance_id   = RenderableManager::Get().GetInstanceID(entity);

//         UnorderedMap<VertexAttributesBitmask, const MeshDataHandle*> bitmask_2_mesh_data;

//         // [Instance Level]: fill instance data
//         auto& inst_data                   = m_cpu_scene.instance_datas[instance_id];
//         inst_data.first_geom_idx          = geom_array[0].geom_id;
//         inst_data.geom_cnt                = mesh_obj->geom_cnt;
//         inst_data.first_geom_instance_idx = geom_instance_cnt;
//         inst_data.model2world             = TransformManager::Get().Get(entity).GetMatrix3x4();

//         inst_data.prev_model2world = inst_data.model2world;
//         inst_data.padding          = 0;

//         // [Geometry Level]: fill geometry instance data
//         uint geom_local_idx = 0;
//         for (auto& geom : geom_array) {
//             // uint               vtx_offset = geo->local_vtx_offset + info.vtx_offset;
//             uint  vtx_offset = geom.local_vtx_offset;// FIXME
//             auto& geo_data   = m_cpu_scene.geometry_datas[geom.geom_id];

//             const auto& mesh_data_handle = mesh_data_handle_array[geom.mesh_data_id];
//             const auto& cpu_mesh_data    = *mesh_data_handle.cpu_mesh_data;
//             const auto& gpu_mesh_data    = mesh_data_handle.gpu_mesh_data;

//             bitmask_2_mesh_data[mesh_data_handle.cpu_mesh_data->vertex_factory_buffers.GetAttributesBitmask()] = &mesh_data_handle;

//             geo_data.num_indices          = geom.local_idx_count;
//             geo_data.num_vertices         = geom.local_vtx_count;
//             geo_data.vertex_offset        = vtx_offset * sizeof(float3);
//             geo_data.prev_vertex_offset   = ~0u;
//             geo_data.normal_offset        = cpu_mesh_data.GetAttributeRange(EVertexAttributes::VA_NORMAL).offset + vtx_offset * VertexAttributesTool::GetSize(EVertexAttributes::VA_NORMAL);
//             geo_data.tangent_offset       = cpu_mesh_data.GetAttributeRange(EVertexAttributes::VA_TANGENT).offset + vtx_offset * VertexAttributesTool::GetSize(EVertexAttributes::VA_TANGENT);
//             geo_data.texcoord0_offset     = cpu_mesh_data.GetAttributeRange(EVertexAttributes::VA_TEXCOORD0).offset + vtx_offset * VertexAttributesTool::GetSize(EVertexAttributes::VA_TEXCOORD0);
//             geo_data.texcoord1_offset     = cpu_mesh_data.GetAttributeRange(EVertexAttributes::VA_TEXCOORD1).offset + vtx_offset * VertexAttributesTool::GetSize(EVertexAttributes::VA_TEXCOORD1);
//             geo_data.mat_idx_and_type     = geom.material_id << 8 | (uint)mat_instances[geom_local_idx]->GetMaterial()->GetType();
//             geo_data.index_offset         = geom.local_idx_offset * sizeof(uint);
//             geo_data.index_buffer_handle  = gpu_mesh_data.idx_bdls_handle;
//             geo_data.vertex_buffer_handle = gpu_mesh_data.vtx_bdls_handle;

//             auto& geom_instance        = m_cpu_scene.geom_instances[geom_instance_cnt];
//             geom_instance.instance_idx = instance_id;
//             geom_instance.geom_idx     = geom.geom_id;

//             ++geom_instance_cnt;
//             ++geom_local_idx;
//         }

//         // [Instance Level]: fill instance vtx_view & idx_view
//         {
//             UnorderedMap<VertexAttributesBitmask, Array<Render::VertexBuffer>>& vtx_view = m_cpu_scene.vtx_views[instance_id];
//             UnorderedMap<VertexAttributesBitmask, Render::IndexBuffer>&         idx_view = m_cpu_scene.idx_views[instance_id];

//             for (auto& [bitmask, mesh_data_handle] : bitmask_2_mesh_data) {
//                 Array<Render::VertexBuffer> vtxs;

//                 // 注意，这里vtxs的顺序不能被改变！具体顺序应当由VertexAttributesTool::GetArrayFromBitmask返回的数组决定！
//                 // => 逻辑关联处：RHICommand.h -> MeshDrawData
//                 auto attrs = VertexAttributesTool::GetArrayFromBitmask(bitmask);

//                 vtxs.reserve(attrs.size());

//                 const auto& cpu_mesh_data = *mesh_data_handle->cpu_mesh_data;
//                 const auto& gpu_mesh_data = mesh_data_handle->gpu_mesh_data;

//                 for (auto& attr : attrs) {
//                     vtxs.push_back({gpu_mesh_data.vertex_buffer.Get(), cpu_mesh_data.GetAttributeRange(attr).offset});
//                 }

//                 vtx_view[bitmask] = std::move(vtxs);
//                 idx_view[bitmask] = {gpu_mesh_data.index_buffer->GetView(),
//                                      EIndexElementType::IET_UINT32};
//             }
//         }
//     }

//     /******************* create material textures, build to dx/vulkan images, store (name, tex-handle) to material instance buffer */
//     // Moer::UnorderedMap<std::string, Render::TextureRef> textures;
//     // Moer::Array<TextureBuilder>                         texture_builders;
//     // texture_builders.reserve(_scene_desc.m_textures.size());
//     // for (auto& texture : _scene_desc.m_textures) {
//     //     auto& builder = texture_builders.emplace_back();
//     //     builder.Data(texture.second.data.data(), texture.second.data.size());
//     //     builder.Width(texture.second.width);
//     //     builder.Height(texture.second.height);
//     //     builder.Format(texture.second.format);
//     //     builder.MipAndLayers(texture.second.mips, texture.second.layers, texture.second.mip_offsets.data(), texture.second.mip_extents.data());
//     //     builder.Name(texture.first);
//     // }
//     // textures = TextureBuilder::BuildTexturesInBatch(texture_builders);
//     // Render::Sampler                         sampler(SF_LINEAR, SAM_REPEAT);
//     // UnorderedMap<std::string, SceneTexture> scene_textures;
//     // //  scene_data.m_textures = GpuSceneBufferBuilder::
//     // for (auto& material_instance : _scene_desc.m_material_instances) {

//     //     if (!_scene_desc.m_mat_instance_textures.contains(material_instance->GetName())) {
//     //         continue;
//     //     }
//     //     for (auto& texture : _scene_desc.m_mat_instance_textures[material_instance->GetName()].textures) {
//     //         uint32_t handle = _scene->GetBindlessArray()->AllocateTexture(textures[texture.second]->GetView(0, textures[texture.second]->GetNumMips()), sampler);
//     //         material_instance->SetParameter(texture.first, handle);
//     //         scene_textures[texture.second] = {textures[texture.second], handle};
//     //     }
//     // }
//     // _scene->RegisterMaterialTextures(scene_textures);

//     // Moer::Array<byte> material_data(_scene_desc.m_material_instances.size() * Material::MaterialBytesNum);
//     // for (auto [name, index] : _scene_desc.m_material_instance_indexes) {
//     //     auto& material = _scene_desc.m_material_instances[index];
//     //     memcpy(material_data.data() + index * Material::MaterialBytesNum, material->GetUniformBuffer().GetData(), material->GetUniformBuffer().GetSize());
//     // }

//     // [GPU Resources]: create gpu buffers, import from io queue to copy queue
//     auto instance_data_buffer     = device.CreateBuffer<byte>("Scene::InstanceDataBuffer", GetInstanceDatas().size_bytes(), EBufferUsageFlags::UNORDERED_ACCESS);
//     auto geometry_data_buffer     = device.CreateBuffer<byte>("Scene::GeometryDataBuffer", GetGeometryDatas().size_bytes(), EBufferUsageFlags::UNORDERED_ACCESS);
//     auto geometry_instance_buffer = device.CreateBuffer<byte>("Scene::GeometryInstanceBuffer", GetGeometryInstances().size_bytes(), EBufferUsageFlags::UNORDERED_ACCESS);

//     Array<LightComponentData> lights(m_lights.entities.size());
//     for (uint i = 0; i < m_lights.entities.size(); ++i) {
//         lights[i] = LightComponentManager::Get().Get(m_lights.entities[i])->ToData();
//     }

//     auto light_buffer = device.CreateBuffer<byte>("Scene::LightBuffer", lights.size() * sizeof(LightComponentData), EBufferUsageFlags::UNORDERED_ACCESS);

//     // cmd_list.CopyFrom(
//     //     material_data,
//     //     material_buffer->GetView(),
//     //     "CopyFrom material_buffer");

//     cmd_list.CopyFrom(
//         std::span<byte>((byte*)lights.data(), lights.size() * sizeof(LightComponentData)),
//         light_buffer->GetView(),
//         "CopyFrom light_buffer");

//     cmd_list.CopyFrom(
//         std::span<byte>((byte*)GetInstanceDatas().data(), GetInstanceDatas().size() * sizeof(Render::InstanceData)),
//         instance_data_buffer->GetView(),
//         "CopyFrom instance_data_buffer");

//     cmd_list.CopyFrom(
//         std::span<byte>((byte*)GetGeometryDatas().data(), GetGeometryDatas().size() * sizeof(Render::GeometryData)),
//         geometry_data_buffer->GetView(),
//         "CopyFrom geometry_data_buffer");

//     cmd_list.CopyFrom(
//         std::span<byte>((byte*)GetGeometryInstances().data(), GetGeometryInstances().size() * sizeof(Render::GeometryInstance)),
//         geometry_instance_buffer->GetView(),
//         "CopyFrom geometry_instance_buffer");

//     auto evt = copy_queue.Execute(cmd_list.Submit());
//     copy_queue.Sync(evt.timeline);

//     m_gpu_scene.buffers[(uint32_t)EGpuSceneResource::InstanceInfo]     = instance_data_buffer;
//     m_gpu_scene.buffers[(uint32_t)EGpuSceneResource::GeometryInfo]     = geometry_data_buffer;
//     m_gpu_scene.buffers[(uint32_t)EGpuSceneResource::GeometryInstance] = geometry_instance_buffer;
//     m_gpu_scene.buffers[(uint32_t)EGpuSceneResource::LightInfo]        = light_buffer;
//     // m_gpu_scene.buffers[(uint32_t)EGpuSceneResource::MaterialInfo] = material_buffer;

//     /******************* import material buffers, light buffers, geometry data buffers, instance data buffers from cpu-io queue to the gpu copy queue */
//     EmplaceIOImportedBuffer(instance_data_buffer);
//     EmplaceIOImportedBuffer(geometry_data_buffer);
//     EmplaceIOImportedBuffer(geometry_instance_buffer);
//     EmplaceIOImportedBuffer(light_buffer);
//     // _scene->EmplaceIOImportedBuffer(material_buffer);
// }

/* scene loading state */
AsyncSceneLoadInfoRef Scene::GetCurrentSceneLoadInfo() noexcept {
    return m_load_info;
}

bool Scene::RegisterAsyncLoadInfo(AsyncSceneLoadInfoRef _load_info) {
    if (m_load_info) {
        if (m_load_info.IsValid() && !m_load_info->IsReady()) {
            LOG_ERROR("Scene is already loading");
            return false;
        }
        //TODO: release current_scene
    }
    m_load_info = _load_info;
    return true;
}

void Scene::ResetAsyncLoadInfo() noexcept {
    m_load_info = nullptr;
}

Scene* AsyncSceneLoadInfo::TryGetScene() {
    if (progress.load(std::memory_order_acq_rel) == 1) {
        return scene;
    }
    return nullptr;
}

} // namespace Moer