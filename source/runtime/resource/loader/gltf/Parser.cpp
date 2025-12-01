#include "loader/gltf/Parser.h"

#include "PixelFormat.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "log/LogSystem.h"
#include "math/Base.h"
#include "math/Function.h"
#include "misc/CountableRef.h"
#include "misc/MMemory.h"
#include "misc/STL.h"
#include "rhi/RHICommon.h"
#include "scene/Entity.h"
#include "scene/Scene.h"
#include "scene/light/LightComponent.h"
#include "scene/material/BufferInterfaceBlock.h"
#include "scene/material/Material.h"
#include "scene/material/MaterialInstance.h"


#include "shaderheaders/shared/utils/Packing.h"
#include <cmath>
#include <filesystem>
#include <functional>
#include <meshoptimizer.h>
#include <stb/stb_image.h>

namespace Moer::Resource::Gltf {

using namespace Moer::ECS;

// using GeometrySet = Moer::UnorderedSet<aiMesh*>;

// struct GeomSetHash {
//     size_t operator()(const GeometrySet& _set) const {
//         size_t hash = 0;
//         for (auto& mesh : _set) {
//             hash ^= std::hash<aiMesh*>{}(mesh);
//         }
//         return hash;
//     }
// };

// struct GeomSetEqual {
//     bool operator()(const GeometrySet& _lhs, const GeometrySet& _rhs) const {
//         if (_lhs.size() != _rhs.size()) {
//             return false;
//         }
//         for (auto& mesh : _lhs) {
//             if (_rhs.find(mesh) == _rhs.end()) {
//                 return false;
//             }
//         }
//         return true;
//     }
// };
// using GeometrySet2MeshObject = UnorderedMap<GeometrySet, SharedPtr<MeshObject>, GeomSetHash, GeomSetEqual>;

// struct Counter {
//     uint32_t vertex = 0;
//     uint32_t index  = 0;

//     Counter& operator+=(const Counter& _rhs) {
//         vertex += _rhs.vertex;
//         index += _rhs.index;
//         return *this;
//     }
// };

// VertexAttributesBitmask GetAttributesBitmask(const aiMesh* _mesh) {
//     Moer::Array<EVertexAttributes> attributes;

//     if (_mesh->HasPositions()) {
//         attributes.push_back(EVertexAttributes::VA_POSITION);
//     }
//     if (_mesh->HasNormals()) {
//         attributes.push_back(EVertexAttributes::VA_NORMAL);
//     }
//     if (_mesh->HasTangentsAndBitangents()) {
//         attributes.push_back(EVertexAttributes::VA_TANGENT);
//     }
//     if (_mesh->HasTextureCoords(0)) {
//         attributes.push_back(EVertexAttributes::VA_TEXCOORD0);
//     }
//     if (_mesh->HasTextureCoords(1)) {
//         attributes.push_back(EVertexAttributes::VA_TEXCOORD1);
//     }

//     return VertexAttributesTool::GetBitmaskFromArray(attributes);
// }

static float3 ToFloat3f(const aiVector3D& _vec) {
    return {_vec.x, _vec.y, _vec.z};
}

static float3 ToFloat3f(const aiColor3D& _vec) {
    return {_vec.r, _vec.g, _vec.b};
}

static float4 ToFloat4f(const aiColor4D& _vec) {
    return {_vec.r, _vec.g, _vec.b, _vec.a};
}

struct Parser::Impl {
    bool LoadSceneFromFile(
        const std::filesystem::path& _file_path,
        Scene&                       _scene,
        bool                         _delete_after_load = false
    );

    std::filesystem::path m_file_parent_path{};

    const aiScene*                       m_gltf_scene = nullptr;
    UnorderedMap<std::string, aiCamera*> m_cameras{};
    UnorderedMap<std::string, aiLight*>  m_lights{};
    UnorderedMap<uint64, Entity>         m_meshes{};

protected:
    void LoadCameras(const aiScene* _ai_scene, Scene& _scene);
    void LoadLights(const aiScene* _ai_scene, Scene& _scene);
    void LoadMaterials(const aiScene* _ai_scene, Scene& _scene);

protected:
    void LoadNode(
        const aiNode*                           _node,
        Scene&                                  _scene,
        const Entity                            _parent,
        std::function<Entity(const aiNode*)>&   _on_load_mesh,
        std::function<Entity(const aiCamera*)>& _on_load_camera,
        std::function<Entity(const aiLight*)>&  _on_load_light
    );

    aiCamera* IsCameraNode(const aiNode* _node) const;
    aiLight*  IsLightNode(const aiNode* _node) const;
    uint64    GetMeshKey(const aiNode* _node) const;

    Transform GetTransform(const aiMatrix4x4& _transform) const;

    void LoadHierarchy(
        const aiScene*                                    _scene,
        const aiNode*                                     _node,
        std::function<void(const aiNode*, const Entity)>& _on_load_object
    );
    void LoadTexture(
        const aiScene*       _scene,
        const aiString&      _texture_path,
        MaterialInstanceRef& _mat,
        const std::string&   _param_namee,
        EPixelFormat         _preferred_format = PF_R8G8B8A8_UNORM
    );
};

/**
     * @brief Load scene from a desc file using Assimp.
     * 
     * @param _file_path 
     * @param _scene 
     * @param _delete_after_load 
     */
bool Parser::Impl::LoadSceneFromFile(
    const std::filesystem::path& _file_path,
    Scene&                       _scene,
    bool                         _delete_after_load
) {
    Assimp::Importer importer;
    auto             real_path = std::filesystem::weakly_canonical(_file_path);
    if (!std::filesystem::exists(real_path)) {
        LOG_WARNING("File not exist: {}", real_path.string());
        LOG_WARNING("Please check the `scene_path` in source/configs/MoerEngine.toml.");
        return false;
    }
    const auto* gltf_scene = importer.ReadFile(
        _file_path.string(),
        aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenBoundingBoxes | aiProcess_GenNormals |
            aiProcess_CalcTangentSpace
    );
    if (!gltf_scene) {
        LOG_WARNING("Failed to load gltf file: {} ", _file_path.string());
        return false;
    }

    m_gltf_scene = gltf_scene;

    m_file_parent_path = _file_path.parent_path();
    auto filename      = _file_path.filename().string();

    // preprocess camera and light nodes
    for (uint i = 0; i < gltf_scene->mNumCameras; ++i) {
        auto* camera                     = gltf_scene->mCameras[i];
        m_cameras[camera->mName.C_Str()] = camera;
    }
    for (uint i = 0; i < gltf_scene->mNumLights; ++i) {
        auto* light                    = gltf_scene->mLights[i];
        m_lights[light->mName.C_Str()] = light;
    }

    // root node
    auto root_entity = _scene.AddTransform(filename);

    // materials
    auto load_material = [&]() {
        for (auto i = 0; i < gltf_scene->mNumMaterials; ++i) {
            auto* material = gltf_scene->mMaterials[i];
            auto  name     = material->GetName();
            if (name.length == 0) {
                // Assimp will import a default material with empty name, so we need to assign a default name
                name = aiString("default_material");
            }
            auto mat_entity = _scene.AddMaterial(name.C_Str());
            //TODO: fill material properties maybe
            _scene.Attach(mat_entity, root_entity);
            // auto* mat_component = _scene.materials.GetComponent(mat_entity);
            // mat_component->SetMaterial(material);
        }
    };
    load_material();

    // meshes
    std::function<Entity(const aiNode*)> load_mesh = [&](const aiNode* _node) {
        auto mesh_key = GetMeshKey(_node);
        if (m_meshes.contains(mesh_key)) {
            return m_meshes[mesh_key];
        }
        // per MeshComponent
        auto mesh_entity = _scene.AddMesh(gltf_scene->mMeshes[_node->mMeshes[0]]->mName.C_Str());
        _scene.Attach(mesh_entity, root_entity);
        auto* mesh_component = _scene.meshes.GetComponent(mesh_entity);

        mesh_component->geometries.resize(_node->mNumMeshes);

        uint32 idx_cnt = 0;
        for (uint i = 0; i < _node->mNumMeshes; ++i) {
            auto* ai_mesh = gltf_scene->mMeshes[_node->mMeshes[i]];
            for (uint j = 0; j < ai_mesh->mNumFaces; ++j) {
                idx_cnt += ai_mesh->mFaces[j].mNumIndices;
            }
        }
        mesh_component->indices.resize(idx_cnt);

        uint32 idx_offset = 0;
        uint32 vtx_offset = 0;
        for (auto i = 0; i < _node->mNumMeshes; ++i) {
            auto* ai_mesh = gltf_scene->mMeshes[_node->mMeshes[i]];

            const auto aabb = Box3D{ToFloat3f(ai_mesh->mAABB.mMin), ToFloat3f(ai_mesh->mAABB.mMax)};
            mesh_component->bounding_box.Expand(aabb);

            auto& geometry = mesh_component->geometries[i];
            // geometry.material_entity = _scene.materials.GetEntity(ai_mesh->mMaterialIndex);
            geometry.material_idx = ai_mesh->mMaterialIndex; // MARK: TODO: retrive on update
            geometry.index_offset = idx_offset;

            // indices
            for (auto j = 0; j < ai_mesh->mNumFaces; ++j) {
                const auto& face = ai_mesh->mFaces[j];
                for (auto k = 0; k < face.mNumIndices; ++k) {
                    mesh_component->indices[idx_offset + k] = face.mIndices[k];
                }
                idx_offset += face.mNumIndices;
            }

            geometry.index_count = idx_offset - geometry.index_offset;

            // vertex attributes
            for (auto j = 0; j < ai_mesh->mNumVertices; ++j) {
                mesh_component->positions.emplace_back(ToFloat3f(ai_mesh->mVertices[j]));

                if (ai_mesh->HasNormals()) {
                    mesh_component->normals.emplace_back(ToFloat3f(ai_mesh->mNormals[j]));
                    mesh_component->packed_normals.emplace_back(Pack_Normal(ToFloat3f(ai_mesh->mNormals[j])));
                }
                if (ai_mesh->HasTangentsAndBitangents()) {
                    mesh_component->tangents.emplace_back(ToFloat3f(ai_mesh->mTangents[j]));
                    mesh_component->packed_tangents.emplace_back(Pack_Normal(ToFloat3f(ai_mesh->mBitangents[j]
                    )));
                }
                if (ai_mesh->HasTextureCoords(0)) {
                    mesh_component->uv0.emplace_back(
                        ai_mesh->mTextureCoords[0][j].x, ai_mesh->mTextureCoords[0][j].y
                    );
                }
                if (ai_mesh->HasTextureCoords(1)) {
                    mesh_component->uv1.emplace_back(
                        ai_mesh->mTextureCoords[1][j].x, ai_mesh->mTextureCoords[1][j].y
                    );
                }
                // if (ai_mesh->HasVertexColors(0)) {
                //     mesh_component->vertex_colors[vtx_offset + j] = ToFloat4f(ai_mesh->mColors[0][j]);
                // }
            }
            vtx_offset += ai_mesh->mNumVertices;
        }

        // create mesh render data
        mesh_component->CreateGpuData(_scene.bindless_array);

        m_meshes[mesh_key] = mesh_entity;

        return mesh_entity;
    };
    // cameras
    std::function<Entity(const aiCamera*)> load_camera = [&](const aiCamera* _camera) {
        auto  camera_entity    = _scene.AddCamera(_camera->mName.C_Str());
        auto* camera_component = _scene.cameras.GetComponent(camera_entity);

        // The interpretation of 'mHorizontalFOV' is inconsistent among gltf2, fbx, and the documentation in the official Assimp version (5.4.2).
        // In this project, we ensure 'mHorizontalFOV' is the 'half' of the horizontal field of view angle (at least in GLTF2 and FBX).
        auto full_yfov_deg       = AI_RAD_TO_DEG(2 * atan(tan(_camera->mHorizontalFOV) / _camera->mAspect));
        camera_component->camera = MoerNew(Camera)();
        camera_component->camera->SetProjectionFactor(
            full_yfov_deg, _camera->mAspect, _camera->mClipPlaneNear, _camera->mClipPlaneFar
        );

        return camera_entity;
    };
    // lights
    std::function<Entity(const aiLight*)> load_light = [&](const aiLight* _light) {
        auto  light_entity    = _scene.AddLight(_light->mName.C_Str());
        auto& light_component = *_scene.lights.GetComponent(light_entity);

        switch (_light->mType) {
            case aiLightSourceType::aiLightSource_DIRECTIONAL:
                light_component = MoerNew(DirectionalLightComponent)(
                    ToFloat3f(_light->mColorDiffuse), // color
                    1.0f,                             // intensity
                    ToFloat3f(_light->mDirection),    // direction
                    0.f                               // angular_size
                );
                break;
            case aiLightSourceType::aiLightSource_POINT:
                light_component = MoerNew(PointLightComponent)(
                    ToFloat3f(_light->mColorDiffuse), // color
                    1.0f,                             // intensity
                    ToFloat3f(_light->mPosition)      // position
                );
                break;
            case aiLightSourceType::aiLightSource_SPOT:
                light_component = MoerNew(SpotLightComponent)(
                    ToFloat3f(_light->mColorDiffuse),       // color
                    1.0f,                                   // intensity
                    ToFloat3f(_light->mPosition),           // position
                    ToFloat3f(_light->mDirection),          // direction
                    AI_RAD_TO_DEG(_light->mAngleInnerCone), // inner_cone_angle
                    AI_RAD_TO_DEG(_light->mAngleOuterCone)  // outer_cone_angle
                );
                break;
            case aiLightSourceType::aiLightSource_AMBIENT:
                light_component = MoerNew(AmbientLightComponent)(ToFloat3f(_light->mColorDiffuse));
                break;
            case aiLightSourceType::aiLightSource_AREA:
                LOG_WARNING("Area light is not supported yet.");
                break;
            default:
                break;
        }

        return light_entity;
    };

    LoadNode(gltf_scene->mRootNode, _scene, root_entity, load_mesh, load_camera, load_light);

    if (_delete_after_load) {
        MoerDelete(this);
    }

    return true;

    // lights

    // MARK: Part 0 Generate Meshlet (Deprecated)
    //
    // Meshlet is deprecated in current version, needs to be re-implemented.
    // You can refer to the old implementation in this commit (browse the file in this commit): 937c1dac60c94d602f000e52f927e33b40a19dae

    // MARK: Part 1 Load mesh node
    // The following code obtains these data:
    //   - m_mesh_object_array, m_geometry_object_array, m_transform_array, m_mesh_data_array
    //   - all about materials & textures (m_materials, m_material_instances, m_material_instance_indexes, m_textures)

    // uint total_vtx_cnt = 0;
    // uint total_idx_cnt = 0;

    // MARK: Part 2 Load material & texture
    // auto& material_instance_array = scene_desc->m_material_instance_array;
    // LoadMaterials(gltf_scene, material_instance_array);

    // // MARK: Part 2 Camera&Light
    // auto& camera_array = scene_desc->m_camera_array;
    // auto& light_array  = scene_desc->m_light_array;
    // LoadCameras(gltf_scene, camera_array);
    // LoadLights(gltf_scene, light_array);

    // // MARK: Path
    // scene_desc->m_path = _file_path.string();

    // if (_delete_after_load) {
    //     MoerDelete(this);
    // }
    // return std::move(scene_desc);

    /**
         * Old code:
         * 
         * // Basic information:
         * m_scene_data->m_path = _file_path.string();
         * 
         * // Data obtained from Part 1:
         * m_scene_data->m_meshlet_descs  = std::move(m_meshlet_descs);
         * m_scene_data->m_meshlet_bounds = std::move(m_meshlet_bounds);
         * 
         * // Data obtained from Part 2:
         * m_scene_data->m_instance_infos = std::move(m_instance_infos);
         * m_scene_data->m_mesh_instances = std::move(m_mesh_instances);
         * 
         * m_scene_data->m_mesh_infos      = std::move(m_mesh_infos);
         * m_scene_data->m_mesh_geometries = std::move(m_mesh_geometries);
         * 
         * m_scene_data->m_materials                 = std::move(m_materials);
         * m_scene_data->m_material_instances        = std::move(m_material_instances);
         * m_scene_data->m_material_instance_indexes = std::move(m_material_instance_indexes);
         * m_scene_data->m_mat_instance_textures     = std::move(m_mat_instance_textures);
         * m_scene_data->m_textures                  = std::move(m_textures);
         * 
         * // Data obtained from Part 3:
         * m_scene_data->m_mesh_buffers = std::move(m_mesh_buffers);
         * 
         * // Data obtained from Part 4:
         * m_scene_data->m_cameras = std::move(m_cameras);
         * m_scene_data->m_lights  = std::move(m_lights);
         */
}

// void Parser::Impl::LoadMeshes(const aiScene* _ai_scene, Moer::Scene::Scene& _scene, ECS::Entity _parent) {
//     for (auto i = 0; i < _ai_scene->mNumMeshes; i++) {
//         auto mesh   = _ai_scene->mMeshes[i];
//         auto entity = _scene.AddMesh(mesh->mName.C_Str());
//         _scene.Attach(entity, root_entity, true);
//     }

//     // 预处理，生成Mesh Buffers
//     // 扫描场景中所有的Mesh，记录下所有顶点属性组合，并且创建对应的MeshBuffers(VertexFactoryBuffers)
//     //
//     // 例子：一些Mesh有position+normal+tangent+uv0，而另一些Mesh只有position+normal，那么就会创建两个MeshBuffers(VertexFactoryBuffers)
//     // 在之后的代码中，会直接将Mesh的数据填充到对应的MeshBuffers中
//     GeometrySet2MeshObject geom_record;

//     auto create_empty_mesh_data_array = [&](const aiScene* _scene) {
//         UnorderedMap<VertexAttributesBitmask, Counter> bitmask_2_counter_map;

//         // 预处理出所有不同的顶点属性组合
//         for (uint32_t i = 0; i < _scene->mNumMeshes; i++) {
//             const auto* mesh    = _scene->mMeshes[i];
//             const auto& bitmask = GetAttributesBitmask(mesh);

//             const auto& counter = Counter{mesh->mNumVertices, mesh->mNumFaces * 3};

//             bitmask_2_counter_map[bitmask] += counter;
//         }

//         UnorderedMap<VertexAttributesBitmask, uint> bitmask_2_mesh_data_array_idx_map;

//         // 创建MeshData per bitmask
//         for (const auto& [bitmask, counter] : bitmask_2_counter_map) {
//             // create mesh data
//             auto& mesh_data = _mesh_data_array.emplace_back();

//             // extract array<attr> from bitmask
//             const auto& vertex_attributes = VertexAttributesTool::GetArrayFromBitmask(bitmask);

//             // initialize vertex factory buffers' length
//             mesh_data.vertex_factory_buffers = VertexFactoryBuffers(vertex_attributes);
//             for (const auto& attr : vertex_attributes) {
//                 mesh_data.vertex_factory_buffers.SetBufferLength(attr, counter.vertex);
//             }
//             // initialize index buffer's length
//             mesh_data.indices.resize(counter.index);

//             // 塞进map里
//             bitmask_2_mesh_data_array_idx_map[bitmask] = _mesh_data_array.size() - 1;
//         }

//         return std::make_pair(
//             std::move(bitmask_2_mesh_data_array_idx_map),
//             std::move(bitmask_2_counter_map));
//     };

//     auto [bitmask_2_mesh_data_array_idx_map, bitmask_2_counter_map] = create_empty_mesh_data_array(_scene);

//     LOG_INFO("MeshDataArray creation info:");
//     for (const auto& [bitmask, counter] : bitmask_2_counter_map) {
//         LOG_INFO("\tBitmask: {}, Vertex Count: {}, Index Count: {}, Face Num: {}", bitmask, counter.vertex, counter.index, counter.index / 3);
//     }
//     // 清空counter
//     for (auto& [_, counter] : bitmask_2_counter_map) {
//         counter = Counter{0, 0};
//     }

//     std::function<void(const aiNode*)> process_mesh_node = [&](const aiNode* _node) {
//         // per mesh object
//         if (_node->mMeshes) {
//             GeometrySet geo_set;

//             for (auto i = 0; i < _node->mNumMeshes; ++i) {
//                 auto* mesh = _scene->mMeshes[_node->mMeshes[i]];
//                 geo_set.insert(mesh);
//             }

//             if (!geom_record.contains(geo_set)) {
//                 uint  local_vtx_cnt = 0;
//                 uint  local_idx_cnt = 0;
//                 Box3D bounding_box;

//                 auto& mesh_object = _mesh_object_array.emplace_back(MakeShared<MeshObject>());

//                 mesh_object->name     = _node->mName.C_Str();
//                 mesh_object->geom_id  = _geom_object_array.size();
//                 mesh_object->geom_cnt = _node->mNumMeshes;
//                 // per geometry object
//                 for (auto i = 0; i < _node->mNumMeshes; ++i) {
//                     auto* mesh = _scene->mMeshes[_node->mMeshes[i]];
//                     Box3D current_box{
//                         {mesh->mAABB.mMin.x, mesh->mAABB.mMin.y, mesh->mAABB.mMin.z},
//                         {mesh->mAABB.mMax.x, mesh->mAABB.mMax.y, mesh->mAABB.mMax.z}};
//                     bounding_box.Expand(current_box);
//                     // 1. geometry desc info
//                     auto& geom_object = _geom_object_array.emplace_back();
//                     {
//                         geom_object.local_idx_count = mesh->mNumFaces * 3;
//                         geom_object.local_vtx_count = mesh->mNumVertices;
//                         geom_object.bounding_box    = current_box;
//                         geom_object.geom_id         = _geom_object_array.size() - 1;
//                         geom_object.material_id     = mesh->mMaterialIndex;

//                         // local_idx_cnt += mesh->mNumFaces * 3;
//                         // local_vtx_cnt += mesh->mNumVertices;
//                     }

//                     // 2. create mesh data array
//                     const auto& bitmask       = GetAttributesBitmask(mesh);
//                     auto        mesh_data_idx = bitmask_2_mesh_data_array_idx_map[bitmask];
//                     auto&       mesh_data     = _mesh_data_array[mesh_data_idx];
//                     auto&       vf_buffers    = mesh_data.vertex_factory_buffers;

//                     auto& idx_offset = bitmask_2_counter_map[bitmask].index;
//                     auto& vtx_offset = bitmask_2_counter_map[bitmask].vertex;

//                     if (mesh->HasPositions()) {
//                         auto* position_buffer = reinterpret_cast<VertexAttributesType<EVertexAttributes::VA_POSITION>::type*>(
//                             vf_buffers.GetBufferData(EVertexAttributes::VA_POSITION));
//                         for (uint j = 0; j < mesh->mNumVertices; ++j) {
//                             const auto& pos                 = mesh->mVertices[j];
//                             position_buffer[vtx_offset + j] = {pos.x, pos.y, pos.z};
//                         }
//                     }

//                     if (mesh->HasNormals()) {
//                         auto* normal_buffer = reinterpret_cast<VertexAttributesType<EVertexAttributes::VA_NORMAL>::type*>(
//                             vf_buffers.GetBufferData(EVertexAttributes::VA_NORMAL));
//                         for (uint j = 0; j < mesh->mNumVertices; ++j) {
//                             const auto& nor               = mesh->mNormals[j];
//                             float3      normal            = {nor.x, nor.y, nor.z};
//                             normal_buffer[vtx_offset + j] = Pack_Normal(normal);
//                         }
//                     }

//                     if (mesh->HasTangentsAndBitangents()) {
//                         auto* tangent_buffer = reinterpret_cast<VertexAttributesType<EVertexAttributes::VA_TANGENT>::type*>(
//                             vf_buffers.GetBufferData(EVertexAttributes::VA_TANGENT));
//                         for (uint j = 0; j < mesh->mNumVertices; ++j) {
//                             const auto& tan                = mesh->mTangents[j];
//                             float3      tangent            = {tan.x, tan.y, tan.z};
//                             tangent_buffer[vtx_offset + j] = Pack_Normal(tangent);
//                         }
//                     }

//                     if (mesh->HasTextureCoords(0)) {
//                         auto* texcoord0_buffer = reinterpret_cast<VertexAttributesType<EVertexAttributes::VA_TEXCOORD0>::type*>(
//                             vf_buffers.GetBufferData(EVertexAttributes::VA_TEXCOORD0));
//                         for (uint j = 0; j < mesh->mNumVertices; ++j) {
//                             const auto& uv0                  = mesh->mTextureCoords[0][j];
//                             texcoord0_buffer[vtx_offset + j] = {uv0.x, uv0.y};
//                         }
//                     }

//                     if (mesh->HasTextureCoords(1)) {
//                         auto* texcoord1_buffer = reinterpret_cast<VertexAttributesType<EVertexAttributes::VA_TEXCOORD1>::type*>(
//                             vf_buffers.GetBufferData(EVertexAttributes::VA_TEXCOORD1));
//                         for (uint j = 0; j < mesh->mNumVertices; ++j) {
//                             const auto& uv1                  = mesh->mTextureCoords[1][j];
//                             texcoord1_buffer[vtx_offset + j] = {uv1.x, uv1.y};
//                         }
//                     }

//                     for (uint j = 0; j < mesh->mNumFaces; ++j) {
//                         const auto& face = mesh->mFaces[j];
//                         for (uint k = 0; k < face.mNumIndices; ++k) {
//                             mesh_data.indices[idx_offset + j * 3 + k] = face.mIndices[k];
//                         }
//                     }

//                     geom_object.local_idx_offset = idx_offset;
//                     geom_object.local_vtx_offset = vtx_offset;

//                     idx_offset += mesh->mNumFaces * 3;
//                     vtx_offset += mesh->mNumVertices;
//                 }

//                 mesh_object->bounding_box = bounding_box;

//                 geom_record[geo_set] = mesh_object;

//                 // total_vtx_cnt += local_vtx_cnt;
//                 // total_idx_cnt += local_idx_cnt;
//             }

//             _transform_array.emplace_back(GetTransform(_node).GetMatrix3x4());

//             auto& mesh_instance        = _mesh_instance_array.emplace_back();
//             mesh_instance.instance_id  = _mesh_instance_array.size() - 1;
//             mesh_instance.transform_id = _transform_array.size() - 1;
//             mesh_instance.mesh_id      = geom_record[geo_set]->mesh_id;
//         }
//     };
//     LoadNodes(_scene, _scene->mRootNode, process_mesh_node);
//     // fill mesh data ranges
//     for (auto& mesh_data : _mesh_data_array) {
//         mesh_data.FillRanges();
//     }
// }

// void Parser::Impl::LoadCameras(const aiScene* _ai_scene, Moer::Scene::Scene& _scene) {
//     if (!_scene->HasCameras()) {
//         LOG_INFO("No camera found, create default camera");
//         _camera_array.push_back(Camera::CreateDefaultCamera());
//     } else {
//         const auto camera_num = _scene->mNumLights;
//         LOG_INFO("Found {} cameras in the scene", camera_num);
//         for (auto i = 0; i < camera_num; ++i) {
//             const auto* camera = _scene->mCameras[i];
//             Vector4f    position(camera->mPosition.x, camera->mPosition.y, camera->mPosition.z, 1.f);
//             Vector4f    look_at_vector(camera->mLookAt.x, camera->mLookAt.y, camera->mLookAt.z, 0.f);
//             Vector4f    up(camera->mUp.x, camera->mUp.y, camera->mUp.z, 0.f);
//             auto*       camera_node           = _scene->mRootNode->FindNode(camera->mName);
//             Transform   camera_node_transform = GetTransform(camera_node);

//             Vector4f world_position       = camera_node_transform * position;
//             Vector4f world_look_at_vector = camera_node_transform * look_at_vector;
//             Vector4f world_up             = camera_node_transform * up;
//             Vector4f world_look_at_point  = world_position + world_look_at_vector;

//             CameraRef camera_ref  = MoerNew(Camera)();
//             Transform transform   = Transform();
//             auto      world_2_cam = Transform(Vector3f(world_position), Vector3f(world_look_at_point), Vector3f(world_up));
//             transform.matrix      = Inverse(world_2_cam.GetMatrix4x4());

//             // The interpretation of 'mHorizontalFOV' is inconsistent among gltf2, fbx, and the documentation in the official Assimp version (5.4.2).
//             // In this project, we ensure 'mHorizontalFOV' is the 'half' of the horizontal field of view angle (at least in GLTF2 and FBX).
//             float full_yfov_deg = AI_RAD_TO_DEG(2 * atan(tan(camera->mHorizontalFOV) / camera->mAspect));
//             camera_ref->Initialize(
//                 transform,
//                 full_yfov_deg,
//                 camera->mAspect,
//                 camera->mClipPlaneNear,
//                 camera->mClipPlaneFar);
//             _camera_array.push_back(camera_ref);
//         }
//     }
// }

// /**
//  * Load lights from gltf scene
//  * Refer to: https://assimp-docs.readthedocs.io/en/latest/API/API-Documentation.html#_CPPv47aiLight
//  */
// void Parser::Impl::LoadLights(const aiScene* _ai_scene, Moer::Scene::Scene& _scene) {
//     if (_scene->HasLights()) {
//         LOG_INFO("No lights found, loader will use default lights");
//         _light_array = std::move(LightComponent::CreateDefaultLightComponents());
//     } else {
//         const auto light_num = _scene->mNumLights;
//         LOG_INFO("Found {} lights in the scene", light_num);

//         for (uint32_t i = 0; i < light_num; i++) {
//             const auto* light = _scene->mLights[i];
//             const auto* node  = _scene->mRootNode->FindNode(light->mName);
//             float3x4    model = GetTransform(node).GetMatrix3x4();
//             if (light->mType == aiLightSourceType::aiLightSource_DIRECTIONAL) {
//                 LightComponentRef light_component = MoerNew(DirectionalLightComponent)(
//                     ToVector3f(light->mColorDiffuse),// color
//                     1.0f,                            // intensity
//                     ToVector3f(light->mDirection),   // direction
//                     0.f                              // angular_size
//                 );
//                 _light_array.push_back(light_component);

//             } else if (light->mType == aiLightSourceType::aiLightSource_POINT) {
//                 float3            pos             = model * float4(ToVector3f(light->mPosition), 1.f);
//                 LightComponentRef light_component = MoerNew(PointLightComponent)(
//                     Min(ToVector3f(light->mColorDiffuse), float3(1.f)),// color
//                     1.0f,                                              // intensity
//                     pos                                                // position
//                 );
//                 // LOG_DEBUG("Point Light Position: {}", pos.ToString());
//                 // LOG_DEBUG("Diffuse  Color: {}", ToVector3f(light->mColorDiffuse).ToString());
//                 // LOG_DEBUG("Specular Color: {}", ToVector3f(light->mColorSpecular).ToString());
//                 // LOG_DEBUG("Ambient  Color: {}", ToVector3f(light->mColorAmbient).ToStriWng());
//                 _light_array.push_back(light_component);

//             } else if (light->mType == aiLightSourceType::aiLightSource_SPOT) {
//                 float3            pos             = model * float4(ToVector3f(light->mPosition), 1.f);
//                 float3            dir             = model * float4(ToVector3f(light->mDirection), 0.f);
//                 LightComponentRef light_component = MoerNew(SpotLightComponent)(
//                     ToVector3f(light->mColorDiffuse),// color
//                     1.0f,                            // intensity
//                     pos,                             // position
//                     dir,                             // direction
//                     light->mAngleInnerCone,          // inner_cone_angle
//                     light->mAngleOuterCone           // outer_cone_angle
//                 );
//                 _light_array.push_back(light_component);

//             } else if (light->mType == aiLightSourceType::aiLightSource_AMBIENT) {
//                 LOG_WARNING("Unsupported light type `Ambient Light` in loading scene");

//             } else if (light->mType == aiLightSourceType::aiLightSource_AREA) {
//                 LOG_WARNING("Unsupported light type `Area Light` in loading scene");

//             } else {
//                 LOG_WARNING("Unsupported light type in loading scene. Light type: {}", static_cast<int>(light->mType));
//             }
//         }
//     }
// }

// MaterialRef GetDefaultMaterial() {
//     MaterialBuilder material_builder{};
//     MaterialRef     default_material = MoerNew(Material)();
//     material_builder.SetParameter("base_color_factor", UniformType::FLOAT4);
//     material_builder.SetParameter("emissive_factor", UniformType::FLOAT3);
//     material_builder.SetParameter("metalic_factor", UniformType::FLOAT);
//     material_builder.SetParameter("roughness_factor", UniformType::FLOAT);
//     material_builder.SetParameter("ao", UniformType::FLOAT);

//     material_builder.SetTexture("albedo_map", ETextureDimension::TEX_2D);
//     material_builder.SetTexture("normal_map", ETextureDimension::TEX_2D);
//     material_builder.SetTexture("metallic_roughness_map", ETextureDimension::TEX_2D);
//     material_builder.SetTexture("ao_map", ETextureDimension::TEX_2D);
//     material_builder.SetTexture("emissive_map", ETextureDimension::TEX_2D);
//     material_builder.SetType(EMaterialType::E_PBR_STANDARD);
//     material_builder.SetName("standered");

//     return material_builder.Build();
// }

void Parser::Impl::LoadMaterials(const aiScene* _ai_scene, Scene& _scene) {

    // if (!data->m_materials.contains("standered")) {
    //     data->m_materials["standered"] = GetDefaultMaterial();
    // }
    // const auto material = data->m_materials["standered"];

    // MaterialInstanceRef mi                            = data->m_material_instances.emplace_back(material->CreateInstance());
    // uint                load_idx                      = data->m_material_instances.size() - 1;
    // data->m_material_instance_indexes[_material_name] = load_idx;

    // mi->SetName(_material_name);

    // LOG_DEBUG("Load Material: {}; Index: {}", _material_name, load_idx);

    // /**
    //  * 关于所有的map参数 (albedo_map, normal_map, etc..)
    //  * | texture | factor | map |
    //  * |---------|--------|-----|
    //  * |  true   | true   | >=0 |
    //  * |  true   | false  | >=0 |
    //  * |  false  | true   | -1  |
    //  * |  false  | false  | -2  |
    //  */

    // // MARK: Base Color
    // {
    //     // base color factor
    //     aiColor4D base_color_factor;

    //     if (_ai_material->Get(AI_MATKEY_COLOR_DIFFUSE, base_color_factor) == AI_SUCCESS) {
    //         Vector4f base_color_factor_cast = *reinterpret_cast<Vector4f*>(&base_color_factor);
    //         mi->SetParameter("albedo_map", int(-1));
    //         mi->SetParameter("base_color_factor", base_color_factor_cast);
    //         LOG_DEBUG("\tLoad Base Color Factor: {}", base_color_factor_cast.ToString());
    //     } else {
    //         mi->SetParameter("albedo_map", int(-2));
    //     }

    //     // base color texture
    //     aiString base_color_path;

    //     if (_ai_material->GetTexture(aiTextureType_BASE_COLOR, 0, &base_color_path) == AI_SUCCESS) {
    //         LoadTexture(_ai_scene, base_color_path, mi, "albedo_map");
    //     } else if (_ai_material->GetTexture(aiTextureType_DIFFUSE, 0, &base_color_path) == AI_SUCCESS) {
    //         LoadTexture(_ai_scene, base_color_path, mi, "albedo_map");
    //     }
    // }

    // // MARK: Normal Map
    // {
    //     // normal
    //     mi->SetParameter("normal_map", int(-1));
    //     // -1 means use mesh normals! This default value is important!

    //     // normal texture
    //     aiString normal_path;

    //     if (_ai_material->GetTexture(aiTextureType_NORMALS, 0, &normal_path) == AI_SUCCESS) {
    //         LoadTexture(_ai_scene, normal_path, mi, "normal_map");
    //     } else if (_ai_material->GetTexture(aiTextureType_NORMAL_CAMERA, 0, &normal_path) == AI_SUCCESS) {
    //         LoadTexture(_ai_scene, normal_path, mi, "normal_map");
    //     }
    // }

    // // MARK: Metallic Roughness
    // {
    //     // metallic roughness factor
    //     float metallic_factor  = 0.0;// default value
    //     float roughness_factor = 0.5;// default value

    //     mi->SetParameter("metallic_roughness_map", int(-1));
    //     if (_ai_material->Get(AI_MATKEY_METALLIC_FACTOR, metallic_factor) == AI_SUCCESS) {
    //         mi->SetParameter("metalic_factor", metallic_factor);
    //         LOG_DEBUG("\tLoad Metallic Factor: {}", metallic_factor);
    //     } else {
    //         mi->SetParameter("metalic_factor", metallic_factor);
    //         LOG_DEBUG("\tLoad Metallic Factor (Default Value): {}", metallic_factor);
    //     }
    //     if (_ai_material->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness_factor) == AI_SUCCESS) {
    //         mi->SetParameter("roughness_factor", roughness_factor);
    //         LOG_DEBUG("\tLoad Roughness Factor: {}", roughness_factor);
    //     } else {
    //         mi->SetParameter("roughness_factor", roughness_factor);
    //         LOG_DEBUG("\tLoad Roughness Factor (Default Value): {}", roughness_factor);
    //     }

    //     // metallic roughness texture
    //     aiString metallic_roughness_path;

    //     if (_ai_material->GetTexture(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE, &metallic_roughness_path) == AI_SUCCESS) {
    //         LoadTexture(_ai_scene, metallic_roughness_path, mi, "metallic_roughness_map");
    //     }
    // }

    // // MARK: Emissive
    // {
    //     // emissive factor
    //     aiColor3D emissive_factor;

    //     if (_ai_material->Get(AI_MATKEY_COLOR_EMISSIVE, emissive_factor) == AI_SUCCESS) {
    //         Vector3f emissive_factor_cast = *reinterpret_cast<Vector3f*>(&emissive_factor);
    //         mi->SetParameter("emissive_map", int(-1));
    //         mi->SetParameter("emissive_factor", emissive_factor_cast);
    //         LOG_DEBUG("\tLoad Emissive Factor: {}", emissive_factor_cast.ToString());
    //     } else {
    //         mi->SetParameter("emissive_map", int(-2));
    //     }

    //     // emissive texture
    //     aiString emissive_path;

    //     if (_ai_material->GetTexture(aiTextureType_EMISSION_COLOR, 0, &emissive_path) == AI_SUCCESS) {
    //         LoadTexture(_ai_scene, emissive_path, mi, "emissive_map");
    //     } else if (_ai_material->GetTexture(aiTextureType_EMISSIVE, 0, &emissive_path) == AI_SUCCESS) {
    //         LoadTexture(_ai_scene, emissive_path, mi, "emissive_map");
    //     }
    // }

    // // MARK: AO
    // {
    //     // ao factor
    //     mi->SetParameter("ao_map", int(-1));
    //     mi->SetParameter("ao", 1.0f);
    //     LOG_DEBUG("\tLoad AO Factor (Default Value): 1.0");

    //     // ao texture
    //     aiString ao_path;

    //     if (_ai_material->GetTexture(aiTextureType_AMBIENT_OCCLUSION, 0, &ao_path) == AI_SUCCESS) {
    //         LoadTexture(_ai_scene, ao_path, mi, "ao_map");
    //     } else if (_ai_material->GetTexture(aiTextureType_LIGHTMAP, 0, &ao_path) == AI_SUCCESS) {
    //         LoadTexture(_ai_scene, ao_path, mi, "ao_map");
    //     }
    // }
}

int32_t GetEmbeddedTextureId(const aiString& _path) {
    const char* path_str = _path.C_Str();
    if (_path.length >= 2 && path_str[0] == '*') {
        for (int i = 1; i < _path.length; i++) {
            if (!isdigit(path_str[i])) {
                return -1;
            }
        }
        return std::atoi(path_str + 1); // NOLINT
    }
    return -1;
}

void Parser::Impl::LoadNode(
    const aiNode*                           _node,
    Scene&                                  _scene,
    const Entity                            _parent,
    std::function<Entity(const aiNode*)>&   _on_load_mesh,
    std::function<Entity(const aiCamera*)>& _on_load_camera,
    std::function<Entity(const aiLight*)>&  _on_load_light
) {
    auto entity = ECS::invalid_entity;

    if (_node->mMeshes) {
        entity                     = _scene.AddMeshInstance(_node->mName.C_Str());
        auto* mesh_instance        = _scene.mesh_instances.GetComponent(entity);
        mesh_instance->mesh_entity = _on_load_mesh(_node);
    } else if (auto* camera = IsCameraNode(_node)) {
        entity = _on_load_camera(camera);
    } else if (auto* light = IsLightNode(_node)) {
        entity = _on_load_light(light);
    } else {
        entity = _scene.AddTransform(_node->mName.C_Str());
    }

    auto* transform      = _scene.transforms.GetComponent(entity);
    transform->transform = GetTransform(_node->mTransformation);

    if (_parent != ECS::invalid_entity) {
        _scene.Attach(entity, _parent, true);
    }

    for (uint32_t i = 0; i < _node->mNumChildren; i++) {
        LoadNode(_node->mChildren[i], _scene, entity, _on_load_mesh, _on_load_camera, _on_load_light);
    }
}

aiCamera* Parser::Impl::IsCameraNode(const aiNode* _node) const {
    if (m_cameras.contains(_node->mName.C_Str())) {
        return m_cameras.at(_node->mName.C_Str());
    }
    return nullptr;
}

aiLight* Parser::Impl::IsLightNode(const aiNode* _node) const {
    if (m_lights.contains(_node->mName.C_Str())) {
        return m_lights.at(_node->mName.C_Str());
    }
    return nullptr;
}

uint64 Parser::Impl::GetMeshKey(const aiNode* _node) const {
    if (_node->mNumMeshes == 0) {
        return 0;
    }

    uint64 mesh_key = GetHash(_node->mMeshes[0]);
    for (auto i = 1; i < _node->mNumMeshes; i++) {
        HashCombine(mesh_key, GetHash(_node->mMeshes[i]));
    }

    return mesh_key;
}

Transform Parser::Impl::GetTransform(const aiMatrix4x4& _transform) const {
    Matrix4x4f matrix;
    for (uint32_t i = 0; i < 4; i++) {
        for (uint32_t j = 0; j < 4; j++) {
            matrix[i][j] = _transform[i][j];
        }
    }
    return matrix;
}

void Parser::Impl::LoadTexture(
    const aiScene*       _scene,
    const aiString&      _texture_path,
    MaterialInstanceRef& _matt,
    const std::string&   _param_namee,
    EPixelFormat         _preferred_format
) {
    // if (data->m_textures.contains(texture_path.C_Str())) {
    //     auto texture = data->m_textures[texture_path.C_Str()];
    //     data->m_mat_instance_textures[mat->GetName()].textures.push_back({param_name, texture_path.C_Str()});
    //     SamplerParams params{};
    //     params.max_mip_level = texture.mips;
    //     LOG_DEBUG("\tLoad Texture, texture is already loaded {} for {}", texture_path.C_Str(), param_name);
    //     return;
    // }

    // int32_t         embedded_id = GetEmbeddedTextureId(texture_path);
    // TextureBuilder* builder     = MoerNew(TextureBuilder);
    // ImageReadDesc   image_desc;

    // if (embedded_id >= 0) {
    //     const aiTexture* texture = scene->mTextures[embedded_id];
    //     image_desc               = ImageIO::ReadFromMemory(reinterpret_cast<unsigned char*>(texture->pcData), texture->mWidth * texture->mHeight * 4);
    //     //todo
    // } else {
    //     std::filesystem::path texture_file_path = m_file_parent_path / texture_path.C_Str();
    //     image_desc                              = ImageIO::ReadFromFile(texture_file_path, 4, _preferred_format);
    // }

    // if (!image_desc.IsValid()) {
    //     LOG_WARNING("Load Texture Failed {} for {}", texture_path.C_Str(), param_name);
    //     return;
    // }

    // TextureData texture_data;
    // texture_data.mips      = image_desc.mips;
    // texture_data.layers    = image_desc.layers;
    // texture_data.width     = image_desc.width;
    // texture_data.height    = image_desc.height;
    // texture_data.channal   = image_desc.channal;
    // texture_data.format    = image_desc.format;
    // texture_data.data_size = image_desc.data_size;
    // texture_data.data.resize(image_desc.data_size);
    // std::copy_n(reinterpret_cast<uint8_t*>(image_desc.data), image_desc.data_size, texture_data.data.data());
    // texture_data.mip_offsets = image_desc.mip_offsets;
    // texture_data.mip_extents = image_desc.mip_extents;

    // if (image_desc.data_callback != nullptr) {
    //     image_desc.data_callback(image_desc.data);
    // }

    // data->m_textures[texture_path.C_Str()] = texture_data;
    // data->m_mat_instance_textures[mat->GetName()].textures.push_back({param_name, texture_path.C_Str()});
    // LOG_DEBUG("\tLoad Texture Success {} for {}", texture_path.C_Str(), param_name);
}

bool Parser::LoadSceneFromFile(const std::filesystem::path& _file_path, Scene& _scene) noexcept {
    Impl impl;
    return impl.LoadSceneFromFile(_file_path.generic_string().data(), _scene);
}

} // namespace Moer::Resource::Gltf