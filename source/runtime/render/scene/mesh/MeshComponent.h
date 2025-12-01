#pragma once

#include "math/Function.h"
#include "misc/STL.h"
#include "misc/Traits.h"
#include "rhi/RHIResource.h"
#include "scene/Entity.h"

namespace Moer::ECS {

struct Box3D {
    float3 min;
    float3 max;

    float3 GetCenter() const noexcept {
        return (min + max) * 0.5f;
    }
    float3 GetExtent() const noexcept {
        return max - min;
    }

    void Expand(const float3& _point) noexcept {
        min = Min(min, _point);
        max = Max(max, _point);
    }
    void Expand(const Box3D& _box) noexcept {
        min = Min(min, _box.min);
        max = Max(max, _box.max);
    }
};

struct RENDER_API MeshComponent {
    struct Geometry {
        uint32 index_offset = 0;
        uint32 index_count  = 0;
        // uint32 vertex_offset = 0;
        // uint32 vertex_count  = 0;

        ECS::Entity material_entity = ECS::invalid_entity;

        // Non-serialized attributes:
        uint32 material_idx = 0; // index of the material in the scene material component array
    };

    // basics
    Array<uint32> indices;
    Array<float3> positions;
    Array<float3> normals;
    Array<uint32> packed_normals;
    Array<float3> tangents;
    Array<uint32> packed_tangents;
    Array<float2> uv0;
    Array<float2> uv1;

    // bones

    // additions
    Array<uint32> vertex_colors;

    Array<Geometry> geometries;
    Box3D           bounding_box;

    Render::BufferRef index_buffer;  // index buffer
    Render::BufferRef vertex_buffer; // all static vertex buffers
    uint32            vtx_bdls_handle;
    uint32            idx_bdls_handle;
    // basics
    Render::BufferView ib_view;
    Render::BufferView vb_position_view;
    Render::BufferView vb_normal_view;
    Render::BufferView vb_tangent_view;
    Render::BufferView vb_uv0_view;
    Render::BufferView vb_uv1_view;

    // bones

    // additions
    Render::BufferView vb_color_view;

    void CreateGpuData(Render::BindlessArrayRef _bindless_array);

    // void Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri);
};

} // namespace Moer::ECS