#pragma once
#include "Component.h"
#include "Entity.h"
#include "TextureLibrary.h"
#include "misc/CountableRef.h"
#include "misc/STL.h"
#include "rhi/RHIResource.h"
#include "scene/camera/CameraComponent.h"
#include "scene/instance/MeshInstanceComponent.h"
#include "scene/light/LightComponent.h"
#include "scene/material/MaterialComponent.h"
#include "scene/mesh/MeshComponent.h"
#include "scene/transform/TransformComponent.h"
#include "serialize/Serializer.h"
#include "shaderheaders/shared/Geometry.h"

namespace Moer {
using namespace Moer::ECS;

enum EGpuSceneResource : uint8 {
    VertexBuffer,
    IndexBuffer,
    InstanceBuffer,
    MaterialBuffer,
    TextureBuffer,
    Num
};

struct SceneTexture {
    Render::TextureRef texture;
    uint               bindless_handle;
};

struct EnvMapResource {
    Render::TextureRef texture;
    uint               bindless_handle;
    ECS::Entity        entity;
};

//////////////////////////////////////////////////////////////////////////
// scene loading state
//////////////////////////////////////////////////////////////////////////
struct AsyncSceneLoadInfo {
    bool IsValid() const noexcept {
        return b_valid;
    };
    bool IsReady() const noexcept {
        return b_valid && progress == 1.f;
    };
    float GetProgress() const noexcept {
        return progress.load();
    };
    class Scene* TryGetScene();
    COUNTABLE_IMPLEMENTATION_AUTO_DESTROY
    // private:
    Scene*           scene;
    std::atomic_uint progress = 0u;
    bool             b_valid  = false;
};

using AsyncSceneLoadInfoRef = CountableRef<AsyncSceneLoadInfo>;

struct CpuSceneDesc {
    Array<Render::GeometryData>     geometry_datas;
    Array<Render::InstanceData>     instance_datas;
    Array<Render::GeometryInstance> geom_instances;
};

struct GpuSceneDesc {
    StaticArray<Render::BufferRef, (uint32)EGpuSceneResource::Num> buffers;
    UnorderedMap<std::string, SceneTexture>                        textures;
};

class RENDER_API Scene {
public:
    /* ECS-Entity & Component*/
    Scene(Render::BindlessArrayRef _bindless_array = nullptr) noexcept : bindless_array(_bindless_array) {};
    ~Scene() noexcept = default;
    bool   IsDescendant(Entity _entity, Entity _ancestor) const;
    Entity AddTransform(std::string_view _name);
    Entity AddCamera(std::string_view _name);
    Entity AddLight(std::string_view _name);
    Entity AddMesh(std::string_view _name);
    Entity AddMeshInstance(std::string_view _name);
    Entity AddMaterial(std::string_view _name);
    void   Remove(Entity _entity, bool _recursive = true, bool _keep_sorted = false);

    // Attaches an entity to a parent:
    //	child_already_in_local_space	:	child won't be transformed from world space to local space
    void Attach(Entity _entity, Entity _parent, bool _child_already_in_local_space = false);
    // Detaches the entity from its parent (if it is attached):
    void Detach(Entity _entity);
    // Detaches all children from an entity (if there are any):
    void DetachChildren(Entity _parent);
    // Bakes the transform's current world matrix back to hierarchy local space (if it is part of a hierarchy)
    void TransformWorldToHierarchy(Entity _entity);

    void GatherChildren(Entity _parent, Array<Entity>& _children) const;

    uint           GetEntityCount() const noexcept;
    EnvMapResource GetCurrentEnvMap() const noexcept;
    void           SetCurrentEnvMap(EnvMapResource _env_map);

protected:
    UnorderedMap<Entity, Array<Entity>> m_topdown_hierarchy;
    ComponentLibrary                    m_component_library;
    TextureLibrary                      m_texture_library;

public:
    Render::BindlessArrayRef bindless_array;

    ComponentManager<NameComponent>& names =
        m_component_library.Register<NameComponent>("Scene::NameComponents");
    ComponentManager<LayerComponent>& layers =
        m_component_library.Register<LayerComponent>("Scene::LayerComponents");
    ComponentManager<HierarchyComponent>& hierarchy =
        m_component_library.Register<HierarchyComponent>("Scene::HierarchyComponents");
    ComponentManager<TransformComponent>& transforms =
        m_component_library.Register<TransformComponent>("Scene::TransformComponents");
    ComponentManager<CameraComponent>& cameras =
        m_component_library.Register<CameraComponent>("Scene::CameraComponents");
    ComponentManager<LightComponentRef>& lights =
        m_component_library.Register<LightComponentRef>("Scene::LightComponents");
    ComponentManager<MeshComponent>& meshes =
        m_component_library.Register<MeshComponent>("Scene::MeshComponents");
    ComponentManager<MeshInstanceComponent>& mesh_instances =
        m_component_library.Register<MeshInstanceComponent>("Scene::MeshInstanceComponents");
    ComponentManager<MaterialComponent>& materials =
        m_component_library.Register<MaterialComponent>("Scene::MaterialComponents");

public:
    void Info();
    /* ECS-System tick*/
    void TickTransform();
    void TickHierarchy();
    void TickMesh();
    void TickMaterial();
    void TickMeshInstance();
    void TickCamera();
    void TickLight();

    void Tick();

public:
    // cpu data
    std::span<const Render::GeometryData>     GetGeometryDatas() const noexcept;
    std::span<const Render::InstanceData>     GetInstanceDatas() const noexcept;
    std::span<const Render::GeometryInstance> GetGeometryInstances() const noexcept;
    // gpu data
    // Render::BufferRef GetBuffer(EGpuSceneResource _type) const noexcept;

    void RegisterMaterialTextures(UnorderedMap<std::string, SceneTexture> _textures) noexcept;

    Render::BindlessArrayRef GetBindlessArray() const noexcept;
    GpuSceneDesc&            GetGpuScene() noexcept;

    std::span<const Render::BufferRef> GetIOPendingBuffers() const noexcept;
    void                               ClearIOPendingBuffers() noexcept;

protected:
    friend class SceneCache;
    void EmplaceIOImportedBuffer(Render::BufferRef _buffer);

    // gpu data
    // void SetBuffer(EGpuSceneResource _type, Render::BufferRef _buffer) noexcept;

    // void FillSceneDesc(std::span<const GeometryObject> _geometry_object_array, std::span<const MeshData> _mesh_data_array);

public:
    AsyncSceneLoadInfoRef GetCurrentSceneLoadInfo() noexcept;
    bool                  RegisterAsyncLoadInfo(AsyncSceneLoadInfoRef _load_info);
    void                  ResetAsyncLoadInfo() noexcept;

protected:
    // CPU Scene
    CpuSceneDesc m_cpu_scene;
    // GPU Scene
    GpuSceneDesc m_gpu_scene;

    EnvMapResource cur_env_map{};

    Array<Render::BufferRef> io_pending_buffers;

protected:
    AsyncSceneLoadInfoRef m_load_info{nullptr};
};

} // namespace Moer