#include "MaterialComponent.h"
#include "misc/Hash.h"

namespace Moer::ECS {

PackedMaterialData LitMaterialComponent::CreateRenderData(bool _force_recreate) {
    PackedMaterialData data = {};
    data.packed_0           = float4(base_color_factor.xyz, base_color);
    data.packed_1           = float4(emissive_factor, emissive);
    data.packed_2           = float4(metallic_factor, metallic, roughness_factor, roughness);
    data.packed_3           = float4(reflectance_factor, reflectance, normal, occlusion);
    // TODO: Extra properties are not packed for now
    return data;
}

PackedMaterialData UnlitMaterialComponent::CreateRenderData(bool _force_recreate) {
    PackedMaterialData data = {};
    data.packed_0           = float4(base_color_factor.xyz, base_color);
    data.packed_1           = float4(emissive_factor, emissive);

    return data;
}

PackedMaterialData SubsurfaceMaterialComponent::CreateRenderData(bool _force_recreate) {
    PackedMaterialData data = {};
    data.packed_0           = float4(base_color_factor.xyz, base_color);
    data.packed_1           = float4(emissive_factor, emissive);
    data.packed_2           = float4(metallic_factor, metallic, roughness_factor, roughness);
    data.packed_3           = float4(reflectance_factor, reflectance, normal, occlusion);
    data.packed_4           = float4(subsurface_factor, subsurface, transmission_factor, transmission);
    data.packed_5           = float4(absorption, ior);
    data.packed_6           = float4(thickness, micro_thickness, 0.0f, 0.0f);

    return data;
}

PackedMaterialData ClothMaterialComponent::CreateRenderData(bool _force_recreate) {
    PackedMaterialData data = {};
    data.packed_0           = float4(base_color_factor.xyz, base_color);
    data.packed_1           = float4(emissive_factor, emissive);
    data.packed_2           = float4(metallic_factor, metallic, roughness_factor, roughness);
    data.packed_3           = float4(reflectance_factor, reflectance, normal, occlusion);
    data.packed_4           = float4(sheen_color, subsurface, transmission_factor, transmission);
    data.packed_5           = float4(absorption, ior);
    data.packed_6           = float4(thickness, micro_thickness, 0.0f, 0.0f);

    return data;
}

template<typename T>
    requires std::derived_from<T, MaterialComponent>
MaterialComponentRef MaterialComponentFactory::CreateMaterialComponent(std::string_view _name) {
    MaterialComponentRef material = nullptr;
    if constexpr (std::is_same_v<T, LitMaterialComponent>) {
        material = new LitMaterialComponent(_name.data());
    } else if constexpr (std::is_same_v<T, UnlitMaterialComponent>) {
        material = new UnlitMaterialComponent(_name.data());
    } else if constexpr (std::is_same_v<T, SubsurfaceMaterialComponent>) {
        material = new SubsurfaceMaterialComponent(_name.data());
    } else if constexpr (std::is_same_v<T, ClothMaterialComponent>) {
        material = new ClothMaterialComponent(_name.data());
    } else {
        static_assert(std::false_type::value, "Unsupported MaterialComponent type");
    }

    // compute hash key from name and store by hash
    m_materials[GetHash(_name)] = material;
    return material;
}

MaterialComponentRef MaterialComponentFactory::GetMaterialComponentByName(std::string_view _name) {
    auto it = m_materials.find(GetHash(_name));
    if (it != m_materials.end())
        return it->second;
    return nullptr;
}

UnorderedMap<uint32, MaterialComponentRef> MaterialComponentFactory::m_materials;

} // namespace Moer::ECS