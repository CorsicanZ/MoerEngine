#pragma once
#include "RenderAPI.h"
#include "misc/CountableRef.h"
#include "misc/EnumBitOperation.h"
#include "misc/Singleton.h"
#include "misc/Traits.h"

namespace Moer::ECS {

enum class EMaterialFlags : uint16 {
    NONE            = 0,
    DIRTY           = 1 << 0,
    TWO_SIDED       = 1 << 1,
    CAST_SHADOW     = 1 << 2,
    USE_VERTEXCOLOR = 1 << 3,
    USE_WPO         = 1 << 4,
    OUTLINE         = 1 << 5,
    REFRACTION      = 1 << 6,
    // ...
};

ENUM_BIT_OP_IMPL(EMaterialFlags, FLAG)

enum class EMaterialType : uint8 {
    LIT,
    UNLIT,
    SUBSURFACE,
    CLOTH,
    TYPE_NUM
};

enum class EMaterialBlendMode : uint8 {
    OPAQUE,
    MASKED,
    TRANSLUCENT,
    MODE_NUM
};

static constexpr uint32 material_invalid_id = ~0u;

struct PackedMaterialData {
    float4 packed_0;
    float4 packed_1;
    float4 packed_2;
    float4 packed_3;
    float4 packed_4;
    float4 packed_5;
    float4 packed_6;
    float4 packed_7;
};

class RENDER_API MaterialComponent : public CountableResource {
public:
    static constexpr uint32 packed_material_data_bytes = sizeof(PackedMaterialData);
    inline static const StaticArray<std::string, uint8(EMaterialType::TYPE_NUM)> material_typestr = {
        "LIT",
        "UNLIT",
        "SUBSURFACE",
        "CLOTH",
    };

    MaterialComponent(
        const std::string& _name,
        EMaterialFlags     _flags,
        EMaterialType      _type,
        EMaterialBlendMode _blend_mode
    ) noexcept :
        name(_name),
        flags(_flags),
        type(_type),
        blend_mode(_blend_mode) {};
    ~MaterialComponent();

    virtual bool
    LoadFromAssimpMaterial(class aiMaterial* _ai_material, class TextureComponentFactory& _texture_factory);
    virtual PackedMaterialData CreateRenderData(bool _force_recreate = false) = 0;

    inline bool IsTwoSided() const {
        return (flags & EMaterialFlags::TWO_SIDED) == EMaterialFlags::TWO_SIDED;
    }
    inline bool IsCastingShadow() const {
        return (flags & EMaterialFlags::CAST_SHADOW) == EMaterialFlags::CAST_SHADOW;
    }
    inline bool IsUsingVertexColors() const {
        return (flags & EMaterialFlags::USE_VERTEXCOLOR) == EMaterialFlags::USE_VERTEXCOLOR;
    }
    inline bool IsUsingWind() const {
        return (flags & EMaterialFlags::USE_WPO) == EMaterialFlags::USE_WPO;
    }
    inline bool IsOutlineEnabled() const {
        return (flags & EMaterialFlags::OUTLINE) == EMaterialFlags::OUTLINE;
    }
    inline bool IsRefractionEnabled() const {
        return (flags & EMaterialFlags::REFRACTION) == EMaterialFlags::REFRACTION;
    }

    inline bool IsAlphaTestEnabled() const {
        return blend_mode == EMaterialBlendMode::MASKED;
    }

protected:
    // Properties
    std::string        name;
    EMaterialFlags     flags      = EMaterialFlags::CAST_SHADOW;
    EMaterialType      type       = EMaterialType::LIT;
    EMaterialBlendMode blend_mode = EMaterialBlendMode::OPAQUE;

    COUNTABLE_DESTROY
};

using MaterialComponentRef = CountableRef<MaterialComponent>;

/** Physical material texture properties
 * 
 * | texture | factor | map                  |
 * |---------|--------|----------------------|
 * |  true   | true   | bindless_handle      |
 * |  true   | false  | bindless_handle      |
 * |  false  | true   | PROPERTY_ONLY_FACTOR |
 * |  false  | false  | PROPERTY_NONE        |
 * 
 *  All color values are linear unless noted otherwise.
 */

class RENDER_API LitMaterialComponent final : public MaterialComponent {
public:
    LitMaterialComponent(
        const std::string& _name,
        EMaterialFlags     _flags      = EMaterialFlags::CAST_SHADOW,
        EMaterialBlendMode _blend_mode = EMaterialBlendMode::OPAQUE
    ) noexcept :
        MaterialComponent(_name, _flags, EMaterialType::LIT, _blend_mode) {}

    PackedMaterialData CreateRenderData(bool _force_recreate = false) override;

protected:
    float4 base_color_factor = {1.0f, 1.0f, 1.0f, 1.0f};
    uint32 base_color        = material_invalid_id; // default: float4(1.0)
    float3 emissive_factor   = {1.0f, 1.0f, 1.0f};
    uint32 emissive          = material_invalid_id; // default: float3(0.0, 0.0, 0.0)

    float  metallic_factor    = 1.0f;
    uint32 metallic           = material_invalid_id; // default 0.0 for non-metals
    float  roughness_factor   = 1.0f;
    uint32 roughness          = material_invalid_id; // default 1.0 for non-metals
    float  reflectance_factor = 1.0f;                // [0..1], prefer > 0.35 for non-metals
    uint32 reflectance        = material_invalid_id; // default: 0.5

    uint32 normal    = material_invalid_id; // default points along +Z (encoded), float3(0.0, 0.0, 1.0)
    uint32 occlusion = material_invalid_id; // default: 0.0

    // Extra material properties for advanced materials
    float3 sheen_color_factor     = {1.0f, 1.0f, 1.0f};
    uint32 sheen_color            = material_invalid_id; // default: float3(0.0, 0.0, 0.0)
    float  sheen_roughness_factor = 1.0f;
    uint32 sheen_roughness        = material_invalid_id; // default: 0.0

    float  clearcoat_factor           = 1.0f;
    uint32 clearcoat                  = material_invalid_id; // default: 1.0
    float  clearcoat_roughness_factor = 1.0f;
    uint32 clearcoat_roughness        = material_invalid_id; // default: 0.0
    // normal for clear coat layer, default points along +Z, float3(0.0, 0.0, 1.0)
    uint32 clear_coat_normal = material_invalid_id;

    float  anisotropy_strength  = 0.0f;
    float3 anisotropy_direction = {1.0f, 0.0f, 0.0f}; // default: float3(1.0, 0.0, 0.0)
    uint32 anisotropy           = material_invalid_id;

    // Refraction properties
    float  transmission_factor = 1.0f;
    uint32 transmission        = material_invalid_id; // default: 1.0
    float3 absorption          = {0.0f, 0.0f, 0.0f};
    float  ior                 = 1.5f;
    float  thickness           = 0.5f;
    float  micro_thickness     = 0.0f;
};

class RENDER_API UnlitMaterialComponent final : public MaterialComponent {
public:
    UnlitMaterialComponent(
        const std::string& _name,
        EMaterialFlags     _flags      = EMaterialFlags::CAST_SHADOW,
        EMaterialBlendMode _blend_mode = EMaterialBlendMode::OPAQUE
    ) noexcept :
        MaterialComponent(_name, _flags, EMaterialType::UNLIT, _blend_mode) {}

    PackedMaterialData CreateRenderData(bool _force_recreate = false) override;

protected:
    float4 base_color_factor = {1.0f, 1.0f, 1.0f, 1.0f};
    uint32 base_color        = material_invalid_id; // default: float4(1.0)
    float3 emissive_factor   = {1.0f, 1.0f, 1.0f};
    uint32 emissive          = material_invalid_id; // default: float3(0.0, 0.0, 0.0)
};

class RENDER_API SubsurfaceMaterialComponent final : public MaterialComponent {
public:
    SubsurfaceMaterialComponent(
        const std::string& _name,
        EMaterialFlags     _flags      = EMaterialFlags::CAST_SHADOW,
        EMaterialBlendMode _blend_mode = EMaterialBlendMode::OPAQUE
    ) noexcept :
        MaterialComponent(_name, _flags, EMaterialType::SUBSURFACE, _blend_mode) {}

    PackedMaterialData CreateRenderData(bool _force_recreate = false) override;

protected:
    float4 base_color_factor = {1.0f, 1.0f, 1.0f, 1.0f};
    uint32 base_color        = material_invalid_id; // default: float4(1.0)
    float3 emissive_factor   = {1.0f, 1.0f, 1.0f};
    uint32 emissive          = material_invalid_id; // default: float3(0.0, 0.0, 0.0)

    float  metallic_factor    = 1.0f;
    uint32 metallic           = material_invalid_id; // default 0.0 for non-metals
    float  roughness_factor   = 1.0f;
    uint32 roughness          = material_invalid_id; // default 1.0 for non-metals
    float  reflectance_factor = 1.0f;                // [0..1], prefer > 0.35 for non-metals
    uint32 reflectance        = material_invalid_id; // default: 0.5

    uint32 normal    = material_invalid_id; // default points along +Z (encoded), float3(0.0, 0.0, 1.0)
    uint32 occlusion = material_invalid_id; // default: 0.0

    // Subsurface scattering properties
    float  subsurface_factor = 12.234f;
    uint32 subsurface        = material_invalid_id; // default: float3(1.0)

    // Refraction properties
    float  transmission_factor = 1.0f;
    uint32 transmission        = material_invalid_id; // default: 1.0
    float3 absorption          = {0.0f, 0.0f, 0.0f};
    float  ior                 = 1.5f;
    float  thickness           = 0.5f;
    float  micro_thickness     = 0.0f;
};

class RENDER_API ClothMaterialComponent final : public MaterialComponent {
public:
    ClothMaterialComponent(
        const std::string& _name,
        EMaterialFlags     _flags      = EMaterialFlags::CAST_SHADOW,
        EMaterialBlendMode _blend_mode = EMaterialBlendMode::OPAQUE
    ) noexcept :
        MaterialComponent(_name, _flags, EMaterialType::CLOTH, _blend_mode) {}

    PackedMaterialData CreateRenderData(bool _force_recreate = false) override;

protected:
    float4 base_color_factor = {1.0f, 1.0f, 1.0f, 1.0f};
    uint32 base_color        = material_invalid_id; // default: float4(1.0)
    float3 emissive_factor   = {1.0f, 1.0f, 1.0f};
    uint32 emissive          = material_invalid_id; // default: float3(0.0, 0.0, 0.0)

    float  metallic_factor    = 1.0f;
    uint32 metallic           = material_invalid_id; // default 0.0 for non-metals
    float  roughness_factor   = 1.0f;
    uint32 roughness          = material_invalid_id; // default 1.0 for non-metals
    float  reflectance_factor = 1.0f;                // [0..1], prefer > 0.35 for non-metals
    uint32 reflectance        = material_invalid_id; // default: 0.5

    uint32 normal    = material_invalid_id; // default points along +Z (encoded), float3(0.0, 0.0, 1.0)
    uint32 occlusion = material_invalid_id; // default: 0.0

    // Cloth properties
    uint32 sheen_color = material_invalid_id; // default: sqrt(base_color)
    uint32 subsurface  = material_invalid_id; // default: float3(0.0)

    // Refraction properties
    float  transmission_factor = 1.0f;
    uint32 transmission        = material_invalid_id; // default: 1.0
    float3 absorption          = {0.0f, 0.0f, 0.0f};
    float  ior                 = 1.5f;
    float  thickness           = 0.5f;
    float  micro_thickness     = 0.0f;
};

// struct [[deprecated("ECS: MaterialComponent is not used yet")]] MaterialComponent {
//     uint8_t user_stencil_ref = 0;

//     enum TEXTURESLOT {
//         BASECOLORMAP,
//         NORMALMAP,
//         SURFACEMAP,
//         EMISSIVEMAP,
//         DISPLACEMENTMAP,
//         OCCLUSIONMAP,
//         TRANSMISSIONMAP,
//         SHEENCOLORMAP,
//         SHEENROUGHNESSMAP,
//         CLEARCOATMAP,
//         CLEARCOATROUGHNESSMAP,
//         CLEARCOATNORMALMAP,
//         SPECULARMAP,
//         ANISOTROPYMAP,
//         TRANSPARENCYMAP,

//         TEXTURESLOT_COUNT
//     };
//     struct TextureMap {
//         std::string        name;
//         Render::TextureRef resource;
//         uint32_t           uvset = 0;
//         // Non-serialized attributes:
//         float lod_clamp                      = 0;  // optional, can be used by texture streaming
//         int   sparse_residencymap_descriptor = -1; // optional, can be used by texture streaming
//         int   sparse_feedbackmap_descriptor  = -1; // optional, can be used by texture streaming
//     };
//     TextureMap textures[TEXTURESLOT_COUNT];

//     int   custom_shader_id = -1;
//     uint4 userdata         = uint4(0, 0, 0, 0); // can be accessed by custom shader

//     // Non-serialized attributes:
//     uint32_t layer_mask         = ~0u;
//     int      sampler_descriptor = -1; // optional

//     // User stencil value can be in range [0, 15]
//     inline void SetUserStencilRef(uint8_t _value) {
//         assert(_value < 16);
//         user_stencil_ref = _value & 0x0F;
//     }
//     uint32_t GetStencilRef() const;

//     inline float GetOpacity() const {
//         return base_color.w;
//     }
//     inline float GetEmissiveStrength() const {
//         return emissive_color.w;
//     }
//     inline int GetCustomShaderID() const {
//         return custom_shader_id;
//     }

//     inline bool HasPlanarReflection() const {
//         return shader_type == SHADERTYPE_PBR_PLANARREFLECTION || shader_type == SHADERTYPE_WATER;
//     }

//     inline void SetDirty(bool _value = true) {
//         if (_value) {
//             flags |= DIRTY;
//         } else {
//             flags &= ~DIRTY;
//         }
//     }
//     inline bool IsDirty() const {
//         return flags & DIRTY;
//     }

//     inline void SetCastShadow(bool _value) {
//         SetDirty();
//         if (_value) {
//             flags |= CAST_SHADOW;
//         } else {
//             flags &= ~CAST_SHADOW;
//         }
//     }
//     inline void SetReceiveShadow(bool _value) {
//         SetDirty();
//         if (_value) {
//             flags &= ~DISABLE_RECEIVE_SHADOW;
//         } else {
//             flags |= DISABLE_RECEIVE_SHADOW;
//         }
//     }
//     inline void SetOcclusionEnabledPrimary(bool _value) {
//         SetDirty();
//         if (_value) {
//             flags |= OCCLUSION_PRIMARY;
//         } else {
//             flags &= ~OCCLUSION_PRIMARY;
//         }
//     }
//     inline void SetOcclusionEnabledSecondary(bool _value) {
//         SetDirty();
//         if (_value) {
//             flags |= OCCLUSION_SECONDARY;
//         } else {
//             flags &= ~OCCLUSION_SECONDARY;
//         }
//     }

//     // inline wi::enums::BLENDMODE GetBlendMode() const { if (userBlendMode == wi::enums::BLENDMODE_OPAQUE && (GetFilterMask() & wi::enums::FILTER_TRANSPARENT)) return wi::enums::BLENDMODE_ALPHA; else return userBlendMode; }
//     inline bool IsCastingShadow() const {
//         return flags & CAST_SHADOW;
//     }
//     inline bool IsAlphaTestEnabled() const {
//         return alpha_ref <= 1.0f - 1.0f / 256.0f;
//     }
//     inline bool IsUsingVertexColors() const {
//         return flags & USE_VERTEXCOLORS;
//     }
//     inline bool IsUsingWind() const {
//         return flags & USE_WIND;
//     }
//     inline bool IsReceiveShadow() const {
//         return (flags & DISABLE_RECEIVE_SHADOW) == 0;
//     }
//     inline bool IsUsingSpecularGlossinessWorkflow() const {
//         return flags & SPECULAR_GLOSSINESS_WORKFLOW;
//     }
//     inline bool IsOcclusionEnabledPrimary() const {
//         return flags & OCCLUSION_PRIMARY;
//     }
//     inline bool IsOcclusionEnabledSecondary() const {
//         return flags & OCCLUSION_SECONDARY;
//     }
//     inline bool IsCustomShader() const {
//         return custom_shader_id >= 0;
//     }
//     inline bool IsDoubleSided() const {
//         return flags & DOUBLE_SIDED;
//     }
//     inline bool IsOutlineEnabled() const {
//         return flags & OUTLINE;
//     }
//     inline bool IsPreferUncompressedTexturesEnabled() const {
//         return flags & PREFER_UNCOMPRESSED_TEXTURES;
//     }
//     inline bool IsVertexAODisabled() const {
//         return flags & DISABLE_VERTEXAO;
//     }

//     inline void SetBaseColor(const float4& value) {
//         SetDirty();
//         base_color = value;
//     }
//     inline void SetSpecularColor(const float4& value) {
//         SetDirty();
//         specular_color = value;
//     }
//     inline void SetEmissiveColor(const float4& value) {
//         SetDirty();
//         emissive_color = value;
//     }
//     inline void SetRoughness(float value) {
//         SetDirty();
//         roughness = value;
//     }
//     inline void SetReflectance(float value) {
//         SetDirty();
//         reflectance = value;
//     }
//     inline void SetMetalness(float value) {
//         SetDirty();
//         metalness = value;
//     }
//     inline void SetEmissiveStrength(float value) {
//         SetDirty();
//         emissive_color.w = value;
//     }
//     inline void SetTransmissionAmount(float value) {
//         SetDirty();
//         transmission = value;
//     }
//     inline void SetRefractionAmount(float value) {
//         SetDirty();
//         refraction = value;
//     }
//     inline void SetNormalMapStrength(float value) {
//         SetDirty();
//         normal_map_strength = value;
//     }
//     inline void SetParallaxOcclusionMapping(float value) {
//         SetDirty();
//         parallax_occlusion_mapping = value;
//     }
//     inline void SetDisplacementMapping(float value) {
//         SetDirty();
//         displacement_mapping = value;
//     }
//     inline void SetSubsurfaceScatteringColor(float3 value) {
//         SetDirty();
//         subsurface_scattering.x = value.x;
//         subsurface_scattering.y = value.y;
//         subsurface_scattering.z = value.z;
//     }
//     inline void SetSubsurfaceScatteringAmount(float value) {
//         SetDirty();
//         subsurface_scattering.w = value;
//     }
//     inline void SetOpacity(float value) {
//         SetDirty();
//         base_color.w = value;
//     }
//     inline void SetAlphaRef(float value) {
//         SetDirty();
//         alpha_ref = value;
//     }
//     inline void SetUseVertexColors(bool value) {
//         SetDirty();
//         if (value) {
//             flags |= USE_VERTEXCOLORS;
//         } else {
//             flags &= ~USE_VERTEXCOLORS;
//         }
//     }
//     inline void SetUseWind(bool value) {
//         SetDirty();
//         if (value) {
//             flags |= USE_WIND;
//         } else {
//             flags &= ~USE_WIND;
//         }
//     }
//     inline void SetUseSpecularGlossinessWorkflow(bool value) {
//         SetDirty();
//         if (value) {
//             flags |= SPECULAR_GLOSSINESS_WORKFLOW;
//         } else {
//             flags &= ~SPECULAR_GLOSSINESS_WORKFLOW;
//         }
//     }
//     inline void SetSheenColor(const float3& value) {
//         sheen_color = float4(value.x, value.y, value.z, sheen_color.w);
//         SetDirty();
//     }
//     inline void SetSheenRoughness(float value) {
//         sheen_roughness = value;
//         SetDirty();
//     }
//     inline void SetClearcoatFactor(float value) {
//         clearcoat = value;
//         SetDirty();
//     }
//     inline void SetClearcoatRoughness(float value) {
//         clearcoat_roughness = value;
//         SetDirty();
//     }
//     inline void SetCustomShaderID(int id) {
//         custom_shader_id = id;
//     }
//     inline void DisableCustomShader() {
//         custom_shader_id = -1;
//     }
//     inline void SetDoubleSided(bool value = true) {
//         if (value) {
//             flags |= DOUBLE_SIDED;
//         } else {
//             flags &= ~DOUBLE_SIDED;
//         }
//     }
//     inline void SetOutlineEnabled(bool value = true) {
//         if (value) {
//             flags |= OUTLINE;
//         } else {
//             flags &= ~OUTLINE;
//         }
//     }
//     inline void SetPreferUncompressedTexturesEnabled(bool value = true) {
//         if (value) {
//             flags |= PREFER_UNCOMPRESSED_TEXTURES;
//         } else {
//             flags &= ~PREFER_UNCOMPRESSED_TEXTURES;
//         }
//         CreateRenderData(true);
//     }
//     inline void SetVertexAODisabled(bool value = true) {
//         if (value) {
//             flags |= DISABLE_VERTEXAO;
//         } else {
//             flags &= ~DISABLE_VERTEXAO;
//         }
//     }

//     // The MaterialComponent will be written to ShaderMaterial (a struct that is optimized for GPU use)
//     Moer::StaticArray<char, MaterialComponent::MaterialBytesNum> WriteShaderMaterial() const;
//     void WriteShaderTextureSlot(int slot, int descriptor);

//     // Retrieve the array of textures from the material

//     // Returns the bitwise OR of all the wi::enums::FILTER flags applicable to this material
//     uint32_t GetFilterMask() const;

//     // Create texture resources for GPU
// };

class MaterialComponentFactory : public Singleton<MaterialComponentFactory> {
public:
    RENDER_API MaterialComponentFactory();

    template<typename T>
        requires std::derived_from<T, MaterialComponent>
    static RENDER_API MaterialComponentRef CreateMaterialComponent(std::string_view _name);

    // Lookup by name (uses same hash function as storage)
    static RENDER_API MaterialComponentRef GetMaterialComponentByName(std::string_view _name);

private:
    // Use hashed key for material lookup to avoid expensive string keys at runtime
    static UnorderedMap<uint32, MaterialComponentRef> m_materials;
};

} // namespace Moer::ECS