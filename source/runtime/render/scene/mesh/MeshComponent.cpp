#include "scene/mesh/MeshComponent.h"
#include "rhi/RHI.h"

#include "shaderheaders/shared/utils/Packing.h"

namespace Moer::ECS {
void MeshComponent::CreateGpuData(Render::BindlessArrayRef _bindless_array) {
    // Generate tangents if not provided
    if (tangents.empty() && !uv0.empty() && !uv1.empty()) {
        tangents.resize(positions.size());
        packed_tangents.resize(positions.size());

        for (const auto& geometry : geometries) {
            const auto index_offset = geometry.index_offset;
            const auto index_count  = geometry.index_count;

            for (uint32 i = 0; i < index_count; i += 3) {
                const auto i0 = indices[index_offset + i + 0];
                const auto i1 = indices[index_offset + i + 1];
                const auto i2 = indices[index_offset + i + 2];

                const auto& p0 = positions[i0];
                const auto& p1 = positions[i1];
                const auto& p2 = positions[i2];

                const auto& u0 = uv0[i0];
                const auto& u1 = uv0[i1];
                const auto& u2 = uv0[i2];

                const auto& n0 = normals[i0];
                const auto& n1 = normals[i1];
                const auto& n2 = normals[i2];

                // Calculate tangents and bitangents
                const auto face_normal = Moer::Normalizef(n0 + n1 + n2);

                const float x1 = p1.x - p0.x;
                const float x2 = p2.x - p0.x;
                const float y1 = p1.y - p0.y;
                const float y2 = p2.y - p0.y;
                const float z1 = p1.z - p0.z;
                const float z2 = p2.z - p0.z;

                const float s1 = u1.x - u0.x;
                const float s2 = u2.x - u0.x;
                const float t1 = u1.y - u0.y;
                const float t2 = u2.y - u0.y;

                const float  r    = 1.0f / (s1 * t2 - s2 * t1);
                const float3 sdir = {
                    (t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r
                };
                const float3 tdir = {
                    (s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r
                };

                const auto tangent = Moer::Normalizef(sdir - face_normal * Moer::Dot(face_normal, sdir));
                // float      sign    = Moer::Dot(Moer::Cross(tangent, face_normal), tdir) < 0.0f ? -1.0f : 1.0f;

                // Store the tangents and bitangents in the arrays
                tangents[i0]        = tangent;
                tangents[i1]        = tangent;
                tangents[i2]        = tangent;
                packed_tangents[i0] = Pack_Normal(tangent);
                packed_tangents[i1] = Pack_Normal(tangent);
                packed_tangents[i2] = Pack_Normal(tangent);
            }
        }
    }

    // Create gpu buffer and buffer views
    auto&               device     = Render::RenderDevice::Get();
    auto&               copy_queue = device.GetCopyQueue();
    Render::CommandList cmd_list{};

    index_buffer = device.CreateBuffer<uint32>(
        "Scene::index_buffer",
        indices.size(),
        EBufferUsageFlags::INDEX_BUFFER | EBufferUsageFlags::ACCELERATION_STRUCTURE |
            EBufferUsageFlags::UNORDERED_ACCESS
    );
    ib_view = index_buffer->GetView(0, indices.size() * sizeof(uint32));
    cmd_list.CopyFrom(
        std::span<byte>((byte*)indices.data(), indices.size() * sizeof(uint32)),
        ib_view,
        "CopyFrom MeshBuffers index_buffer"
    );

    const auto position_size = positions.size() * sizeof(float3);
    const auto normal_size   = normals.size() * sizeof(uint32);
    const auto tangent_size  = tangents.size() * sizeof(uint32);
    const auto uv0_size      = uv0.size() * sizeof(float2);
    const auto uv1_size      = uv1.size() * sizeof(float2);

    const auto vertex_size = position_size + normal_size + tangent_size + uv0_size + uv1_size;

    vertex_buffer = device.CreateBuffer<byte>(
        "Scene::soa_vertex_buffer",
        vertex_size,
        EBufferUsageFlags::VERTEX_BUFFER | EBufferUsageFlags::ACCELERATION_STRUCTURE |
            EBufferUsageFlags::UNORDERED_ACCESS
    );
    vb_position_view = vertex_buffer->GetView(0, position_size);
    vb_normal_view   = vertex_buffer->GetView(position_size, normal_size);
    vb_tangent_view  = vertex_buffer->GetView(position_size + normal_size, tangent_size);
    vb_uv0_view      = vertex_buffer->GetView(position_size + normal_size + tangent_size, uv0_size);
    vb_uv1_view = vertex_buffer->GetView(position_size + normal_size + tangent_size + uv0_size, uv1_size);

    cmd_list.CopyFrom(
        std::span<byte>((byte*)positions.data(), position_size),
        vb_position_view,
        "CopyFrom MeshBuffers position_buffer"
    );
    cmd_list.CopyFrom(
        std::span<byte>((byte*)normals.data(), normal_size),
        vb_normal_view,
        "CopyFrom MeshBuffers normal_buffer"
    );
    cmd_list.CopyFrom(
        std::span<byte>((byte*)tangents.data(), tangent_size),
        vb_tangent_view,
        "CopyFrom MeshBuffers tangent_buffer"
    );
    cmd_list.CopyFrom(
        std::span<byte>((byte*)uv0.data(), uv0_size), vb_uv0_view, "CopyFrom MeshBuffers texcoord0_buffer"
    );
    cmd_list.CopyFrom(
        std::span<byte>((byte*)uv1.data(), uv1_size), vb_uv1_view, "CopyFrom MeshBuffers texcoord1_buffer"
    );

    idx_bdls_handle = _bindless_array->AllocateBuffer(ib_view);
    vtx_bdls_handle = _bindless_array->AllocateBuffer(vertex_buffer->GetView(0, vertex_size));

    auto evt = copy_queue.Execute(cmd_list.Submit());
    copy_queue.Sync(evt.timeline);
}
}; // namespace Moer::ECS